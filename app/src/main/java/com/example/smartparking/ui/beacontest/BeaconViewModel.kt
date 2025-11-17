package com.example.smartparking.ui.beacontest

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.sqrt

data class BeaconData(
    val name: String,
    val address: String,
    val rssiSamples: MutableList<Int> = mutableListOf()
)

class BeaconViewModel : ViewModel() {

    private val aliasByDeviceName: Map<String, String> = mapOf(
        "BeaconA" to "B1",
        "BeaconB" to "B2",
        "BeaconC" to "B3",
        "Beacon03" to "Gate_In",
        "Beacon_Gate_Out" to "Gate_Out",
    )

    private var fingerprint: Map<String, Map<String, Int>> = mapOf(
        "S1" to mapOf("B1" to -87, "B2" to -88, "B3" to -78),
        "S2" to mapOf("B1" to -75, "B2" to -70, "B3" to -79),
        "S3" to mapOf("B1" to -72, "B2" to -60, "B3" to -55),
        "S4" to mapOf("B1" to -50, "B2" to -55, "B3" to -84),
        "S5" to mapOf("B1" to -82, "B2" to -66, "B3" to -74)
    )

    fun setFingerprint(newFp: Map<String, Map<String, Int>>) {
        fingerprint = newFp
    }

    private val _userLocation = MutableStateFlow<String?>(null)
    val userLocation: StateFlow<String?> = _userLocation

    private val _beacons = MutableStateFlow<List<BeaconData>>(emptyList())
    val beacons: StateFlow<List<BeaconData>> = _beacons

    private val _parkingSlotStatus = MutableStateFlow<Map<String, Boolean>>(
        fingerprint.keys.associateWith { false }
    )
    val parkingSlotStatus: StateFlow<Map<String, Boolean>> = _parkingSlotStatus

    private val samplesByAlias: MutableMap<String, MutableList<Int>> = mutableMapOf()

    private val handler = Handler(Looper.getMainLooper())
    private val intervalMs = 5_000L
    private var scanCallback: ScanCallback? = null
    private var lastEmittedSlot: String? = null

    private var lastKnownParkedSlot: String? = null

    private val minDistanceGap = 15.0
    private val noConfidentClearMs = 10_000L // 10 seconds
    private val rssiAbsentThreshold = -95 // if averages are below this, considered "far"

    private var lastConfidentTimestamp = 0L

    companion object {
        private const val TAG = "BLE"
        private const val MISSING_DEFAULT = -100
    }

    init {
        startRollingEstimation()
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return
        val scanner = adapter.bluetoothLeScanner ?: return

        _beacons.value = emptyList()

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        val filters = emptyList<ScanFilter>()

        val cb = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val name = result.scanRecord?.deviceName ?: result.device?.name ?: return

                if (!name.startsWith("Beacon", ignoreCase = true)) return

                val address = result.device.address
                val rssi = result.rssi

                Log.d(TAG, "📡 Terbaca BLE: name=$name, addr=$address, rssi=$rssi")

                val alias = getAliasFor(name)
                if (alias != null) {
                    val list = samplesByAlias.getOrPut(alias) { mutableListOf() }
                    list.add(rssi)
                    Log.i(TAG, "✅ $name → $alias (total=${list.size})")
                } else {
                    Log.w(TAG, "⚠️ Tidak ada alias cocok untuk $name")
                }

                val curr = _beacons.value.toMutableList()
                val exist = curr.find { it.address == address }
                if (exist != null) {
                    exist.rssiSamples.add(rssi)
                } else {
                    curr.add(BeaconData(name = name, address = address, rssiSamples = mutableListOf(rssi)))
                }
                _beacons.value = curr
            }

            override fun onScanFailed(errorCode: Int) {
                Log.e(TAG, "Scan failed: $errorCode")
            }
        }

        scanCallback = cb
        scanner.startScan(filters, settings, cb)
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        val scanner = BluetoothAdapter.getDefaultAdapter()?.bluetoothLeScanner
        scanCallback?.let { scanner?.stopScan(it) }
        scanCallback = null
        Log.d(TAG, "stopScan() called")
    }

    private fun startRollingEstimation() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                try {
                    estimateAndEmit()
                } finally {
                    handler.postDelayed(this, intervalMs)
                }
            }
        }, intervalMs)
    }

    private fun getAliasFor(name: String): String? {
        val lower = name.lowercase()

        return when {
            lower == "beacona" -> "B1"
            lower == "beaconb" -> "B2"
            lower == "beaconc" -> "B3"
            lower == "beacon03" -> "Gate_In"
            lower == "beacon_gate_out" -> "Gate_Out"
            else -> null
        }
    }

    private fun estimateAndEmit() {
        if (fingerprint.isEmpty()) return

        if (samplesByAlias.isEmpty() || samplesByAlias.values.all { it.isEmpty() }) {
            Log.d(TAG, "❌ Tidak ada sampel BLE sama sekali. Deteksi dibatalkan.")
            _userLocation.value = null
            return
        }

        Log.d(TAG, "📦 samplesByAlias=${samplesByAlias.mapValues { it.value.size }}")

        val gateBeacon = samplesByAlias.keys.find {
            it.equals("Gate_In", ignoreCase = true) ||
                    it.equals("Gate_Out", ignoreCase = true)
        }

        if (gateBeacon != null) {
            _userLocation.value = gateBeacon
            Log.d(TAG, "🚪 Detected gate beacon: $gateBeacon")

            if (gateBeacon.equals("Gate_Out", ignoreCase = true)) {
                lastKnownParkedSlot?.let { parkedSlot ->
                    Log.d(TAG, "Gate_Out detected — marking slot $parkedSlot as EMPTY")
                    val updated = _parkingSlotStatus.value.toMutableMap()
                    updated[parkedSlot] = false
                    _parkingSlotStatus.value = updated
                    lastKnownParkedSlot = null
                }
            }

            samplesByAlias.clear()
            lastEmittedSlot = null
            lastConfidentTimestamp = System.currentTimeMillis()
            return
        }

        val means: Map<String, Int> = fingerprint
            .flatMap { it.value.keys }
            .distinct()
            .associateWith { alias ->
                val samples = samplesByAlias[alias]
                if (samples.isNullOrEmpty()) MISSING_DEFAULT else samples.average().toInt()
            }

        val slotAliases = fingerprint.flatMap { it.value.keys }.distinct()
        val slotMeans = slotAliases.map { means[it] ?: MISSING_DEFAULT }

        val allMissingOrWeak = slotMeans.all { it <= rssiAbsentThreshold || it == MISSING_DEFAULT }

        var bestSlot: String? = null
        var bestDist = Double.MAX_VALUE
        var secondBest = Double.MAX_VALUE

        for ((slotId, fp) in fingerprint) {
            var sumSq = 0.0
            for ((alias, fpRssi) in fp) {
                val real = means[alias] ?: MISSING_DEFAULT
                val d = (real - fpRssi).toDouble()
                sumSq += d * d
            }
            val dist = sqrt(sumSq)
            if (dist < bestDist) {
                secondBest = bestDist
                bestDist = dist
                bestSlot = slotId
            } else if (dist < secondBest) {
                secondBest = dist
            }
        }

        val confident = (secondBest - bestDist) >= minDistanceGap

        if (bestSlot != null && confident) {
            lastConfidentTimestamp = System.currentTimeMillis()

            // update user location to this slot
            if (bestSlot != lastEmittedSlot) {
                lastEmittedSlot = bestSlot
                _userLocation.value = bestSlot
                Log.d(TAG, "Detected slot = $bestSlot (dist=$bestDist, gap=${secondBest - bestDist})")
            } else {
                _userLocation.value = bestSlot
            }

            val updated = _parkingSlotStatus.value.toMutableMap()
            updated[bestSlot] = true
            _parkingSlotStatus.value = updated
            lastKnownParkedSlot = bestSlot

        } else {
            val now = System.currentTimeMillis()

            if (allMissingOrWeak) {
                Log.d(TAG, "All slot beacon signals missing/weak -> clearing userLocation")
                _userLocation.value = null
                lastEmittedSlot = null
                if (lastConfidentTimestamp == 0L) lastConfidentTimestamp = now - noConfidentClearMs - 1L
            } else {
                if (lastConfidentTimestamp == 0L) {
                    // no confident seen yet
                    lastConfidentTimestamp = now
                }
                val timeSinceConfident = now - lastConfidentTimestamp
                if (timeSinceConfident >= noConfidentClearMs) {
                    Log.d(TAG, "No confident for ${timeSinceConfident}ms >= threshold -> clearing userLocation")
                    _userLocation.value = null
                    lastEmittedSlot = null
                } else {
                    Log.d(TAG, "No confident yet but within grace period (${timeSinceConfident}ms)")
                }
            }
        }

        Log.d(TAG, "🧩 Fingerprint aliases: ${fingerprint.flatMap { it.value.keys }}")
        Log.d(TAG, "📶 SamplesByAlias keys: ${samplesByAlias.keys}")

        samplesByAlias.clear()
        _beacons.value = emptyList()
    }

    override fun onCleared() {
        super.onCleared()
        stopScan()
        handler.removeCallbacksAndMessages(null)
    }
}

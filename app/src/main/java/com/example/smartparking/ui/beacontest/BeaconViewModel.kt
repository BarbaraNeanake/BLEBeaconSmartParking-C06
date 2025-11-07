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

/**
 * BeaconViewModel – versi lengkap (5 slot: S1..S5).
 * - Scan BLE (tanpa filter dulu) -> kumpulkan RSSI per device
 * - Map nama device -> alias beacon (B1/B2/B3/..)
 * - Hitung mean RSSI per alias -> bandingkan dengan fingerprint -> tentukan slot terbaik
 */
class BeaconViewModel : ViewModel() {

    /** Mapping nama perangkat BLE -> alias beacon (SESUIKAN dengan yang terlihat di Logcat) */
    private val aliasByDeviceName: Map<String, String> = mapOf(
        "Beacon-Entrance" to "B1",
        "Beacon-Hall"     to "B2",
        "Beacon-Lift"     to "B3",
        // Tambah jika punya beacon lain:
        // "Beacon-Corner"   to "B4",
        // "Beacon-Gate"     to "B5",
    )

    
    /** Fingerprint 5 slot (contoh angka dummy). GANTI dengan hasil kalibrasi lapangan. */
    private var fingerprint: Map<String, Map<String, Int>> = mapOf(
        "S1" to mapOf("B1" to -68, "B2" to -77, "B3" to -78),
        "S2" to mapOf("B1" to -75, "B2" to -70, "B3" to -79),
        "S3" to mapOf("B1" to -60, "B2" to -85, "B3" to -72),
        "S4" to mapOf("B1" to -50, "B2" to -55, "B3" to -84),
        "S5" to mapOf("B1" to -82, "B2" to -66, "B3" to -74)
    )

    fun setFingerprint(newFp: Map<String, Map<String, Int>>) {
        fingerprint = newFp
    }

    /** Debug list beacons (buat ditampilkan di UI bila perlu) */
    private val _beacons = MutableStateFlow<List<BeaconData>>(emptyList())
    val beacons: StateFlow<List<BeaconData>> = _beacons

    /** Hasil estimasi slot terkini: "S1".."S5" (atau null jika belum confident) */
    private val _detectedSlot = MutableStateFlow<String?>(null)
    val detectedSlot: StateFlow<String?> = _detectedSlot

    /** Buffer RSSI untuk tiap alias beacon (rolling window 10 detik) */
    private val samplesByAlias: MutableMap<String, MutableList<Int>> = mutableMapOf()

    private val handler = Handler(Looper.getMainLooper())
    private val intervalMs = 10_000L
    private var scanCallback: ScanCallback? = null
    private var lastEmittedSlot: String? = null

    /** Minimum gap jarak (Euclidean) antara juara 1 & 2 agar dianggap “yakin” */
    private val minDistanceGap = 0.5

    companion object {
        private const val TAG = "BLE"
        private const val MISSING_DEFAULT = -100
    }

    init {
        // Jadwalkan estimasi pertama setelah interval, lalu berulang
//        startRollingEstimation()
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return
        val scanner = adapter.bluetoothLeScanner ?: return

        // kosongkan buffer lama
        _beacons.value = emptyList()

        // (opsional) set ke low-latency biar cepat nangkap
        val settings = android.bluetooth.le.ScanSettings.Builder()
            .setScanMode(android.bluetooth.le.ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        // Tidak pakai ScanFilter karena kita mau prefix, bukan exact match
        val filters = emptyList<android.bluetooth.le.ScanFilter>()

        val cb = object : android.bluetooth.le.ScanCallback() {
            override fun onScanResult(callbackType: Int, result: android.bluetooth.le.ScanResult) {
                // Ambil nama dari iklan dulu, kalau null baru dari device
                val name = result.scanRecord?.deviceName ?: result.device.name ?: return

                // FILTER: hanya perangkat dengan nama diawali "Beacon" (tanpa peduli kapital)
                if (!name.startsWith("Beacon", ignoreCase = true)) return

                val address = result.device.address
                val rssi = result.rssi

                // update list beacon yang kita simpan
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
                android.util.Log.e("BLE", "Scan failed: $errorCode")
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

    /** Ambil mean RSSI per alias -> hitung jarak Euclidean ke fingerprint -> pilih slot terbaik */
    private fun estimateAndEmit() {
        if (fingerprint.isEmpty()) return

        // Mean RSSI untuk tiap alias. Kalau kosong, pakai MISSING_DEFAULT.
        val means: Map<String, Int> = fingerprint
            .flatMap { it.value.keys } // semua alias yang dipakai fingerprint
            .distinct()
            .associateWith { alias ->
                val samples = samplesByAlias[alias]
                if (samples.isNullOrEmpty()) MISSING_DEFAULT else samples.average().toInt()
            }

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

        // Terapkan ambang gap supaya tidak sensitif noise
        val confident = (secondBest - bestDist) >= minDistanceGap

        if (bestSlot != null && confident && bestSlot != lastEmittedSlot) {
            lastEmittedSlot = bestSlot
            _detectedSlot.value = bestSlot
            Log.d(TAG, "Detected slot = $bestSlot (dist=$bestDist, gap=${secondBest - bestDist})")
        } else {
            Log.d(TAG, "No confident slot. best=$bestSlot dist=$bestDist gap=${secondBest - bestDist}")
        }

        // Geser jendela: kosongkan buffer agar pengukuran berikutnya fresh
        samplesByAlias.clear()
        // Debug list beacons juga direset supaya isi UI tidak “membengkak” terus
        _beacons.value = emptyList()
    }

    override fun onCleared() {
        super.onCleared()
        stopScan()
        handler.removeCallbacksAndMessages(null)
    }
}

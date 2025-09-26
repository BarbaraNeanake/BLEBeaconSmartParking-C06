package com.example.smartparking.ui.beacontest

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.pow
import kotlin.math.sqrt

data class BeaconData(
    val name: String,
    val address: String,
    val rssiSamples: MutableList<Int> = mutableListOf()
)

class BeaconViewModel : ViewModel() {

    private val _beacons = MutableStateFlow<List<BeaconData>>(emptyList())
    val beacons: StateFlow<List<BeaconData>> = _beacons

    private val _detectedSlot = MutableStateFlow("Belum terdeteksi")
    val detectedSlot: StateFlow<String> = _detectedSlot

    private val handler = Handler(Looper.getMainLooper())
    private val interval: Long = 10_000 // setiap 10 detik

    // Dataset fingerprint (mean RSSI dari tiap slot)
    private val fingerprintDataset = mapOf(
        "Slot A" to mapOf("B1" to -63, "B2" to -81, "B3" to -67),
        "Slot B" to mapOf("B1" to -70, "B2" to -75, "B3" to -80),
        "Slot C" to mapOf("B1" to -60, "B2" to -85, "B3" to -72)
    )

    init {
        startMeanCalculation()
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        val bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
        val scanner = bluetoothAdapter.bluetoothLeScanner

        // Reset list supaya fresh
        _beacons.value = emptyList()

        val callback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val device = result.device
                val name = device.name ?: return

                // Filter: hanya device dengan nama diawali "Beacon"
                if (name.startsWith("Beacon", ignoreCase = true)) {
                    val address = device.address
                    val rssi = result.rssi

                    // Cek apakah beacon sudah ada di list
                    val currentList = _beacons.value.toMutableList()
                    val existing = currentList.find { it.address == address }

                    if (existing != null) {
                        existing.rssiSamples.add(rssi)
                    } else {
                        currentList.add(
                            BeaconData(
                                name = name,
                                address = address,
                                rssiSamples = mutableListOf(rssi)
                            )
                        )
                    }
                    _beacons.value = currentList
                }
            }
        }

        // Mulai scanning
        scanner.startScan(callback)

        // Simpan callback biar bisa stop nanti
        scanCallback = callback
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        val bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
        val scanner = bluetoothAdapter.bluetoothLeScanner
        scanCallback?.let { scanner.stopScan(it) }
        scanCallback = null
        _beacons.value = emptyList()
    }

    // Tambahkan properti di ViewModel:
    private var scanCallback: ScanCallback? = null


    private fun startMeanCalculation() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                calculatePosition()
                handler.postDelayed(this, interval)
            }
        }, interval)
    }

    private fun calculatePosition() {
        if (_beacons.value.isEmpty()) return

        // Hitung mean RSSI dari beacon real-time
        val meanValues = _beacons.value.associate { beacon ->
            val mean = if (beacon.rssiSamples.isNotEmpty()) {
                beacon.rssiSamples.average().toInt()
            } else {
                -100 // default kalau kosong
            }
            beacon.name to mean
        }

        // Cari slot terdekat dengan Euclidean Distance
        var bestSlot = "Tidak diketahui"
        var bestDistance = Double.MAX_VALUE

        for ((slot, fp) in fingerprintDataset) {
            var sumSq = 0.0
            for ((beaconName, fpRssi) in fp) {
                val realRssi = meanValues[beaconName] ?: -100
                sumSq += (realRssi - fpRssi).toDouble().pow(2.0)
            }
            val distance = sqrt(sumSq)
            if (distance < bestDistance) {
                bestDistance = distance
                bestSlot = slot
            }
        }

        _detectedSlot.value = bestSlot
    }
}

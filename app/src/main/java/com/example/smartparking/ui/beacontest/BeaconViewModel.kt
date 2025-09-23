package com.example.smartparking.ui.beacontest

import android.annotation.SuppressLint
import android.app.Application
import android.bluetooth.BluetoothAdapter
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.os.ParcelUuid
import android.bluetooth.le.ScanFilter
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class BeaconDevice(val name: String?, val address: String)

class BeaconViewModel(application: Application) : AndroidViewModel(application) {

    private val _beacons = MutableStateFlow<List<BeaconDevice>>(emptyList())
    val beacons = _beacons.asStateFlow()

    private val bluetoothAdapter: BluetoothAdapter? =
        BluetoothAdapter.getDefaultAdapter()

    @SuppressLint("MissingPermission")
    fun startScan() {
        val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return

        val filter = ScanFilter.Builder()
            .setServiceUuid(ParcelUuid.fromString("12345678-1234-1234-1234-1234567890ab"))
            .build()

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        scanner.startScan(listOf(filter), settings, object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                viewModelScope.launch {
                    val current = _beacons.value.toMutableList()
                    val exists = current.any { it.address == result.device.address }
                    if (!exists) {
                        current.add(
                            BeaconDevice(
                                name = result.device.name ?: "Unknown",
                                address = result.device.address
                            )
                        )
                        _beacons.value = current
                    }
                }
            }
        })
    }

    // BeaconViewModel.kt
    fun clearBeacons() {
        _beacons.value = emptyList()
    }

}

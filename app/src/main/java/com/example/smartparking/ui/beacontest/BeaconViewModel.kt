package com.example.smartparking.ui.beacontest

import android.os.Handler
import android.os.Looper
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.Parking
import com.example.smartparking.data.repository.ParkingRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import kotlin.math.pow
import kotlin.math.sqrt

data class BeaconData(
    val name: String,
    val address: String,
    val rssiSamples: MutableList<Int> = mutableListOf()
)

class BeaconViewModel(
    private val repository: ParkingRepository
) : ViewModel() {

    companion object {
        private const val DEFAULT_RSSI = -100
        private const val INTERVAL_MS: Long = 10_000
        private const val IOT_URL = "https://danishritonga-caps-backend.hf.space/pelanggaran"
    }

    private val _beacons = MutableStateFlow<List<BeaconData>>(emptyList())
    val beacons: StateFlow<List<BeaconData>> = _beacons

    private val _detectedSlot = MutableStateFlow("Belum terdeteksi")
    val detectedSlot: StateFlow<String> = _detectedSlot

    private val _assignResult = MutableStateFlow("")
    val assignResult: StateFlow<String> = _assignResult

    private val handler = Handler(Looper.getMainLooper())

    private val fingerprintDataset = mapOf(
        1 to mapOf("B1" to -63, "B2" to -81, "B3" to -67),
        2 to mapOf("B1" to -70, "B2" to -75, "B3" to -80),
        3 to mapOf("B1" to -60, "B2" to -85, "B3" to -72)
    )

    init {
        startMeanCalculation()
    }

    fun startScan() {
        // TODO: implementasi BLE scan
        val dummy = listOf(
            BeaconData("B1", "AA:BB:CC", mutableListOf(-69, -71, -70, -70, -69)),
            BeaconData("B2", "DD:EE:FF", mutableListOf(-74, -76, -75, -74, -75)),
            BeaconData("B3", "GG:HH:II", mutableListOf(-81, -79, -80, -80, -79))
        )
        _beacons.value = dummy
    }

    fun stopScan() {
        _beacons.value = emptyList()
    }

    private fun startMeanCalculation() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                calculatePosition()
                handler.postDelayed(this, INTERVAL_MS)
            }
        }, INTERVAL_MS)
    }

    private fun calculatePosition() {
        if (_beacons.value.isEmpty()) return

        val meanValues = _beacons.value.associate { beacon ->
            beacon.name to (beacon.rssiSamples.average().toInt().takeIf { beacon.rssiSamples.isNotEmpty() } ?: DEFAULT_RSSI)
        }

        var bestSlot = "Tidak diketahui"
        var bestDistance = Double.MAX_VALUE

        fingerprintDataset.forEach { (slot, fp) ->
            val distance = fp.entries.sumOf { (name, fpRssi) ->
                val realRssi = meanValues[name] ?: DEFAULT_RSSI
                (realRssi - fpRssi).toDouble().pow(2.0)
            }.let(::sqrt)

            if (distance < bestDistance) {
                bestDistance = distance
                bestSlot = slot.toString()
            }
        }

        _detectedSlot.value = bestSlot
    }

    fun assignSlot(userId: Int, userRole: String) {
        viewModelScope.launch {
            val detected = _detectedSlot.value.toIntOrNull()
            if (detected == null) {
                _assignResult.value = "Slot belum jelas"
                return@launch
            }

            val response = repository.getParkings()
            if (!response.isSuccessful) {
                _assignResult.value = "Gagal ambil data parkir dari server"
                return@launch
            }

            val parkings = response.body() ?: emptyList()
            val slot = parkings.find { it.nomor == detected }

            if (slot == null) {
                _assignResult.value = "Slot $detected tidak ada di database"
                return@launch
            }

            handleSlotAssignment(slot, userId, userRole)
        }
    }

    private suspend fun handleSlotAssignment(slot: Parking, userId: Int, userRole: String) {
        if (slot.status == "false" && slot.rolesUser == userRole) {
            val updatedSlot = slot.copy(userid = userId, status = "True")
            val updateRes = repository.updateParking(slot.nomor, updatedSlot)

            _assignResult.value = if (updateRes.isSuccessful) {
                "Slot ${slot.lokasi} berhasil diassign ke user $userId"
            } else {
                "Gagal update slot di server"
            }
        } else {
            _assignResult.value = "Slot tidak tersedia atau role tidak sesuai"
            sendIoTNotification()
        }
    }

    private fun sendIoTNotification() {
        val client = OkHttpClient()
        val requestBody = """{"sensor_id": "1"}""".toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(IOT_URL).post(requestBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                println("Gagal kirim data ke IoT: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                println("Response dari IoT: ${response.body?.string()}")
            }
        })
    }
}

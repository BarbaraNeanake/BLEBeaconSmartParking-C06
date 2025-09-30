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
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
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

    private val _beacons = MutableStateFlow<List<BeaconData>>(emptyList())
    val beacons: StateFlow<List<BeaconData>> = _beacons

    private val _detectedSlot = MutableStateFlow("Belum terdeteksi")
    val detectedSlot: StateFlow<String> = _detectedSlot

    private val _assignResult = MutableStateFlow<String>("")
    val assignResult: StateFlow<String> = _assignResult

    private val handler = Handler(Looper.getMainLooper())
    private val interval: Long = 10_000 // setiap 10 detik

    // Dataset fingerprint (mean RSSI dari tiap slot)
    private val fingerprintDataset = mapOf(
        1 to mapOf("B1" to -63, "B2" to -81, "B3" to -67),
        2 to mapOf("B1" to -70, "B2" to -75, "B3" to -80),
        3 to mapOf("B1" to -60, "B2" to -85, "B3" to -72)
    )

//    private val slotMapping = mapOf(
//        "Slot A" to 1,
//        "Slot B" to 2,
//        "Slot C" to 3
//    )

    init {
        startMeanCalculation()
    }

    fun startScan() {
        // TODO: tambahkan implementasi scanner BLE kamu di sini
        // sementara dummy data biar jalan
        val dummy = listOf(
            BeaconData("B1", "AA:BB:CC", mutableListOf(-69, -71, -70, -70, -69)), // Rata-rata mendekati -70
            BeaconData("B2", "DD:EE:FF", mutableListOf(-74, -76, -75, -74, -75)), // Rata-rata mendekati -75
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
                bestSlot = slot.toString()
            }
        }

        _detectedSlot.value = bestSlot
    }

    /**
     * Assign slot ke user berdasarkan hasil deteksi beacon
     */
    fun assignSlot(userId: Int, userRole: String) {
        viewModelScope.launch {
            val detected = _detectedSlot.value.toInt()
//            if (detected == "Belum terdeteksi" || detected == "Tidak diketahui") {
//                _assignResult.value = "Slot belum jelas"
//                return@launch
//            }
            println(detected)
            val response = repository.getParkings()
            if (response.isSuccessful) {
                val parkings = response.body() ?: emptyList()

                println(parkings)

                val slot = parkings.find { it.nomor == detected }

                println("🔍 Akan update slot ${slot?.nomor} dengan user $userId")

                if (slot != null) {
                    if (slot.status == "false" && slot.rolesUser == userRole) {
                        val updatedSlot = slot.copy(
                            userid = userId,
                            status = "True"
                        )
                        val updateRes = repository.updateParking(slot.nomor, updatedSlot)

                        _assignResult.value = if (updateRes.isSuccessful) {
                            "Slot ${slot.lokasi} berhasil diassign ke user $userId"
                        } else {
                            "Gagal update slot di server"
                        }
                    } else {
                        _assignResult.value = "Slot tidak tersedia atau role tidak sesuai"

                        val client = OkHttpClient()
                        val requestBody = """{"sensor_id": "1"}""".toRequestBody("application/json".toMediaType())

                        val request = Request.Builder()
                            .url("https://danishritonga-caps-backend.hf.space/pelanggaran") // endpoint untuk IoT
                            .post(requestBody)
                            .build()

                        client.newCall(request).enqueue(object : okhttp3.Callback {
                            override fun onFailure(call: okhttp3.Call, e: IOException) {
                                println("Gagal kirim data ke IoT: ${e.message}")
                            }

                            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                                println("Response dari IoT: ${response.body?.string()}")
                            }
                        })

                    }
                } else {
                    _assignResult.value = "Slot $detected tidak ada di database"
                }
            } else {
                _assignResult.value = "Gagal ambil data parkir dari server"
            }
        }
    }
}

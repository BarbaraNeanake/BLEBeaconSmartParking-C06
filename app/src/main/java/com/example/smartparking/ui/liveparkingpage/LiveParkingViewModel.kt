package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.Parking
import com.example.smartparking.data.remote.ParkingApiService
import com.example.smartparking.data.remote.RetrofitProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlin.collections.*

/**
 * ViewModel ini TIDAK mengatur koordinat slot. Koordinat tetap ditentukan di UI.
 * Ia hanya mengambil 'status' dari backend dan mengekspos peta: slotId -> status.
 */
class LiveParkingViewModel(
    private val api: ParkingApiService = RetrofitProvider.parkingApi
) : ViewModel() {

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    /** Map ID slot -> status (occupied/available/disabled_slot) */
    private val _statusById = MutableStateFlow<Map<String, String>>(emptyMap())
    val statusById: StateFlow<Map<String, String>> = _statusById

    init {
        reload()
    }

    fun reload() {
        if (_loading.value) return
        viewModelScope.launch {
            _loading.value = true
            _error.value = null
            try {
                val resp = api.getParkings()                    // Response<List<Parking>>
                if (!resp.isSuccessful) {
                    throw Exception("HTTP ${resp.code()} ${resp.message()}")
                }

                val rows: List<Parking> = resp.body().orEmpty() // ← sekarang List<Parking>
                val map: Map<String, String> = rows.associate { row ->
                    val id = extractSlotId(row.lokasi) ?: "S${row.nomor}"
                    id to (row.status?.lowercase() ?: "available")
                }

                _statusById.value = map
            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal memuat data"
            } finally {
                _loading.value = false
            }
        }
    }


    private fun extractSlotId(lokasi: String?): String? {
        if (lokasi.isNullOrBlank()) return null
        // cari token seperti S1, S12, D1, dsb.
        val rx = Regex("""\b([SD]\d+|S\d+)\b""", RegexOption.IGNORE_CASE)
        return rx.find(lokasi)?.value?.uppercase()
    }
}
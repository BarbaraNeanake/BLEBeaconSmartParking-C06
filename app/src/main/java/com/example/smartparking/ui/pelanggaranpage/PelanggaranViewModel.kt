package com.example.smartparking.ui.pelanggaranpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class PelanggaranRecord(
    val date: String,
    val slotCode: String,
    val violationType: String
)

data class PelanggaranUiState(
    val loading: Boolean = false,
    val ownerName: String = "Barbara Neanake",
    val records: List<PelanggaranRecord> = emptyList(),
    val error: String? = null
)

class PelanggaranViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(PelanggaranUiState())
    val uiState: StateFlow<PelanggaranUiState> = _uiState

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, error = null)
            try {
                // TODO: ganti ke pemanggilan BE / repo kamu
                delay(350)

                // contoh data dummy — jumlahnya bebas
                val dummy = listOf(
                    PelanggaranRecord("07-11-2025", "P3", "Parkir di luar garis"),
                    PelanggaranRecord("06-11-2025", "P5", "Parkir melebihi waktu"),
                    PelanggaranRecord("05-11-2025", "P12", "Menggunakan slot disabilitas"),
                    PelanggaranRecord("03-11-2025", "P1", "Parkir di area loading"),
                    PelanggaranRecord("01-11-2025", "P7", "Tidak menampilkan identitas kendaraan")
                )

                _uiState.value = _uiState.value.copy(
                    loading = false,
                    records = dummy
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    error = e.message ?: "Gagal memuat data pelanggaran"
                )
            }
        }
    }
}

package com.example.smartparking.ui.pelanggaranpage

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.PelanggaranRepository
import com.example.smartparking.data.repository.dao.SessionDao
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

data class PelanggaranRecord(
    val date: String,
    val slotCode: String,
    val violationType: String
)

data class PelanggaranUiState(
    val loading: Boolean = false,
    val ownerName: String = "",
    val records: List<PelanggaranRecord> = emptyList(),
    val error: String? = null
)

class PelanggaranViewModel(
    private val repo: PelanggaranRepository = PelanggaranRepository(RetrofitProvider.pelanggaranApi),
    private val sessionDao: SessionDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(PelanggaranUiState())
    val uiState: StateFlow<PelanggaranUiState> = _uiState

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, error = null)

            try {
                val session = sessionDao.getSessionOnce()
                val userName = session?.name ?: "Pengguna"
                val userId = session?.userId

                Log.d("PelanggaranVM", "User ID dari session: $userId")

                println("ini userdID ${userId}")

                val resp = if (userId != null)
                    repo.getByUser(userId)
                else
                    repo.getAll()

                println(resp)
                if (resp.code() == 404) {
                    _uiState.value = PelanggaranUiState(
                        loading = false,
                        ownerName = userName,
                        records = emptyList(),
                        error = "Tidak ada pelanggaran yang tercatat."
                    )
                    return@launch
                }

                if (!resp.isSuccessful) throw Exception("HTTP ${resp.code()} ${resp.message()}")

                val records = resp.body().orEmpty().map {
                    val date = it.created_at?.take(19)?.replace("T", " ") ?: "-"
                    PelanggaranRecord(
                        date = date,
                        slotCode = it.nomor?.toString() ?: "-",
                        violationType = it.jenis_pelanggaran ?: "-"
                    )
                }

                _uiState.value = PelanggaranUiState(
                    loading = false,
                    ownerName = userName,
                    records = records
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

class PelanggaranVMFactory(
    private val sessionDao: SessionDao
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(PelanggaranViewModel::class.java)) {
            return PelanggaranViewModel(sessionDao = sessionDao) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}

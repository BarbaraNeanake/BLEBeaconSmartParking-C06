package com.example.smartparking.ui.historypage

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.LogActivity
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.LogActivityRepository
import com.example.smartparking.data.repository.dao.SessionDao
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

data class HistoryUiState(
    val loading: Boolean = true,
    val name: String = "",
    val items: List<LogActivity> = emptyList(),
    val emptyMessage: String? = null,
    val error: String? = null
)

class HistoryViewModel(
    private val logRepository: LogActivityRepository,
    private val sessionDao: SessionDao
) : ViewModel() {

    private val _ui = MutableStateFlow(HistoryUiState())
    val ui: StateFlow<HistoryUiState> = _ui

    init {
        fetchLogs()
    }

    fun retry() = fetchLogs()

    private fun fetchLogs() {
        viewModelScope.launch {
            try {
                _ui.value = _ui.value.copy(loading = true)

                val session = sessionDao.getSessionOnce()
                if (session == null) {
                    _ui.value = HistoryUiState(loading = false, error = "Belum login.")
                    return@launch
                }

                val response = RetrofitProvider.logActivityApi
                    .getLogById(session.userId ?: -1)

                // ✅ Jika API balas 404 (log tidak ditemukan), tampilkan kosong tanpa error
                if (response.code() == 404) {
                    _ui.value = HistoryUiState(
                        loading = false,
                        name = session.name,
                        items = emptyList(),
                        emptyMessage = "Tidak ada log",
                        error = null
                    )
                    return@launch
                }

                if (response.isSuccessful) {
                    val logs = response.body().orEmpty()
                    _ui.value = HistoryUiState(
                        loading = false,
                        name = session.name,
                        items = logs,
                        emptyMessage = if (logs.isEmpty()) "Tidak ada log" else null
                    )
                } else {
                    _ui.value = HistoryUiState(
                        loading = false,
                        error = "Gagal memuat log (${response.code()})"
                    )
                }

            } catch (e: Exception) {
                _ui.value = HistoryUiState(
                    loading = false,
                    error = e.localizedMessage ?: "Terjadi kesalahan tak terduga"
                )
            }
        }
    }
}

/** Manual ViewModelFactory (jika belum pakai Hilt) */
class HistoryViewModelFactory(
    private val logRepository: LogActivityRepository,
    private val sessionDao: SessionDao
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(HistoryViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return HistoryViewModel(logRepository, sessionDao) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}


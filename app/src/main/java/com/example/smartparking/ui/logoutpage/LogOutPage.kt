package com.example.smartparking.ui.logoutpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LogoutUiState(
    val showDialog: Boolean = true,
    val loading: Boolean = false,
    val error: String? = null
)

class LogoutViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(LogoutUiState())
    val uiState: StateFlow<LogoutUiState> = _uiState

    /** Panggil ini saat user menekan tombol Cancel */
    fun cancelDialog() {
        _uiState.update { it.copy(showDialog = false) }
    }

    /**
     * Konfirmasi logout:
     * - set loading
     * - panggil clearSession() (ganti dengan repo DataStore/SharedPref kamu)
     * - callback onSuccess() biar NavGraph pindah ke Login
     */
    fun confirmLogout(onSuccess: () -> Unit) {
        viewModelScope.launch {
            _uiState.update { it.copy(loading = true, error = null) }
            try {
                clearSession()     // TODO ganti dengan AuthRepository.clear()
                onSuccess()
            } catch (e: Exception) {
                _uiState.update { it.copy(error = e.message ?: "Logout gagal") }
            } finally {
                _uiState.update { it.copy(loading = false) }
            }
        }
    }

    /** Dummy: simulasi bersihin sesi. Ganti sesuai implementasi kamu. */
    private suspend fun clearSession() {
        // contoh: delay biar keliatan loading
        delay(400)
        // contoh asli:
        // dataStore.edit { it.remove(TOKEN_KEY) }
        // authRepository.logout()
    }
}

package com.example.smartparking.ui.logoutpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class LogoutUiState(
    val showDialog: Boolean = true,
    val loading: Boolean = false
)

class LogoutViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(LogoutUiState())
    val uiState = _uiState.asStateFlow()

    /** Tutup dialog tanpa logout */
    fun cancelDialog() {
        _uiState.value = _uiState.value.copy(showDialog = false)
    }

    /** Konfirmasi logout (simulasi, nanti ganti call BE) */
    fun confirmLogout(onSuccess: () -> Unit) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true)
            delay(600) // simulasi proses
            _uiState.value = _uiState.value.copy(loading = false, showDialog = false)
            onSuccess()
        }
    }
}

package com.example.smartparking.ui.editpasspage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class EditPassUiState(
    val email: String = "",
    val newPassword: String = "",
    val confirmPassword: String = "",
    val loading: Boolean = false,
    val success: Boolean = false,
    val errorMessage: String? = null
)

class EditPassViewModel : ViewModel() {

    private val _ui = MutableStateFlow(EditPassUiState())
    val uiState: StateFlow<EditPassUiState> = _ui

    fun onEmailChanged(v: String) = _ui.update {
        it.copy(email = v.trim(), errorMessage = null)
    }

    fun onNewPasswordChanged(v: String) = _ui.update {
        it.copy(newPassword = v, errorMessage = null)
    }

    fun onConfirmPasswordChanged(v: String) = _ui.update {
        it.copy(confirmPassword = v, errorMessage = null)
    }

    fun submitReset() {
        val s = _ui.value
        // Validasi sederhana
        if (!s.email.contains("@") || !s.email.contains(".")) {
            _ui.update { it.copy(errorMessage = "Email tidak valid.") }
            return
        }
        if (s.newPassword.length < 6) {
            _ui.update { it.copy(errorMessage = "Password minimal 6 karakter.") }
            return
        }
        if (s.newPassword != s.confirmPassword) {
            _ui.update { it.copy(errorMessage = "Konfirmasi password tidak sama.") }
            return
        }

        viewModelScope.launch {
            _ui.update { it.copy(loading = true, errorMessage = null) }
            // TODO: panggil repository reset password di sini
            delay(800)
            _ui.update { it.copy(loading = false, success = true) }
        }
    }
}

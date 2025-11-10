package com.example.smartparking.ui.editpasspage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.repository.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class EditPassUiState(
    val email: String = "",
    val newPassword: String = "",
    val confirmPassword: String = "",
    val showNew: Boolean = false,
    val showConfirm: Boolean = false,
    val loading: Boolean = false,
    val success: Boolean = false,
    val errorMessage: String? = null
)

class EditPassViewModel(
    private val repo: UserRepository
) : ViewModel() {

    private val _ui = MutableStateFlow(EditPassUiState())
    val uiState: StateFlow<EditPassUiState> = _ui

    fun onEmailChanged(v: String) = _ui.update { it.copy(email = v.trim(), errorMessage = null) }
    fun onNewPasswordChanged(v: String) = _ui.update { it.copy(newPassword = v, errorMessage = null) }
    fun onConfirmPasswordChanged(v: String) = _ui.update { it.copy(confirmPassword = v, errorMessage = null) }

    fun toggleShowNew() = _ui.update { it.copy(showNew = !it.showNew) }
    fun toggleShowConfirm() = _ui.update { it.copy(showConfirm = !it.showConfirm) }

    fun submitReset() {
        val s = _ui.value
        when {
            !s.email.contains("@") || !s.email.contains(".") ->
                _ui.update { it.copy(errorMessage = "Email tidak valid.") }

            s.newPassword.length < 6 ->
                _ui.update { it.copy(errorMessage = "Password minimal 6 karakter.") }

            s.newPassword != s.confirmPassword ->
                _ui.update { it.copy(errorMessage = "Konfirmasi password tidak sama.") }

            else -> {
                viewModelScope.launch {
                    _ui.update { it.copy(loading = true, errorMessage = null) }
                    val result = repo.updatePassword(s.email, s.newPassword)
                    if (result.isSuccess) {
                        _ui.update { it.copy(loading = false, success = true) }
                    } else {
                        val err = result.exceptionOrNull()?.message ?: "Gagal update password"
                        _ui.update { it.copy(loading = false, errorMessage = err) }
                    }

                }
            }
        }
    }
}

package com.example.smartparking.ui.loginpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.repository.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LoginUiState(
    val email: String = "",
    val password: String = "",
    val passwordVisible: Boolean = false,
    val rememberMe: Boolean = false,
    val loading: Boolean = false,
    val errorMessage: String? = null,
    val isLoggedIn: Boolean = false,
    val canSubmit: Boolean = false
)

class LoginViewModel(
    private val userRepository: UserRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState: StateFlow<LoginUiState> = _uiState

    fun onEmailChanged(newEmail: String) = _uiState.update {
        it.copy(
            email = newEmail,
            canSubmit = newEmail.isNotBlank() && it.password.isNotBlank()
        )
    }

    fun onPasswordChanged(newPassword: String) = _uiState.update {
        it.copy(
            password = newPassword,
            canSubmit = it.email.isNotBlank() && newPassword.isNotBlank()
        )
    }

    fun togglePasswordVisibility() = _uiState.update {
        it.copy(passwordVisible = !it.passwordVisible)
    }

    fun onRememberMeChanged(checked: Boolean) = _uiState.update {
        it.copy(rememberMe = checked)
    }

    fun login() {
        val email = _uiState.value.email
        val password = _uiState.value.password

        viewModelScope.launch {
            _uiState.update { it.copy(loading = true, errorMessage = null) }

            val result = userRepository.login(email, password)
            if (result.isSuccess) {
                _uiState.update { it.copy(loading = false, isLoggedIn = true) }
            } else {
                _uiState.update {
                    it.copy(
                        loading = false,
                        errorMessage = result.exceptionOrNull()?.message ?: "Login gagal"
                    )
                }
            }
        }
    }

    fun logout() {
        viewModelScope.launch { userRepository.logout() }
    }
}

class LoginVMFactory(private val repo: UserRepository) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return LoginViewModel(repo) as T
    }
}

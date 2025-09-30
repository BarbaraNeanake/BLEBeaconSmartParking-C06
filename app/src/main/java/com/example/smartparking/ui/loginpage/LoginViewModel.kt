package com.example.smartparking.ui.loginpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class LoginUiState(
    val email: String = "",
    val password: String = "",
    val passwordVisible: Boolean = false,
    val rememberMe: Boolean = false,
    val loading: Boolean = false,
    val errorMessage: String? = null,
    val isLoggedIn: Boolean = false
)

class LoginViewModel(
    private val repo: AuthRepository = FakeAuthRepository() // nanti ganti ke repo asli
) : ViewModel() {

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState = _uiState.asStateFlow()

    fun onEmailChanged(v: String) { _uiState.value = _uiState.value.copy(email = v) }
    fun onPasswordChanged(v: String) { _uiState.value = _uiState.value.copy(password = v) }
    fun togglePasswordVisibility() {
        _uiState.value = _uiState.value.copy(passwordVisible = !_uiState.value.passwordVisible)
    }
    fun onRememberMeChanged(v: Boolean) { _uiState.value = _uiState.value.copy(rememberMe = v) }

    fun login() = viewModelScope.launch {
        val s = _uiState.value
        _uiState.value = s.copy(loading = true, errorMessage = null)

        val result = repo.login(s.email.trim(), s.password)

        _uiState.value = _uiState.value.copy(loading = false)
        result.onSuccess {
            _uiState.value = _uiState.value.copy(isLoggedIn = true)
        }.onFailure { e ->
            _uiState.value = _uiState.value.copy(errorMessage = e.message ?: "Login failed")
        }
    }

    // Placeholder SSO
    fun loginWithGoogle() = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, errorMessage = null)
        delay(600)
        _uiState.value = _uiState.value.copy(loading = false, isLoggedIn = true)
    }
    fun loginWithFacebook() = viewModelScope.launch {
        _uiState.value = _uiState.value.copy(loading = true, errorMessage = null)
        delay(600)
        _uiState.value = _uiState.value.copy(loading = false, isLoggedIn = true)
    }
}

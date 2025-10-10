package com.example.smartparking.ui.loginpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
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
    private val repo: AuthRepository = FakeAuthRepository() // nanti ganti ke repo asli
) : ViewModel() {

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState = _uiState.asStateFlow()

    /* ----------------- Input handlers ----------------- */
    fun onEmailChanged(v: String) = _uiState.update {
        val trimmed = v.trim()
        it.copy(
            email = trimmed,
            errorMessage = null,
            canSubmit = canSubmit(trimmed, it.password)
        )
    }

    fun onPasswordChanged(v: String) = _uiState.update {
        it.copy(
            password = v,
            errorMessage = null,
            canSubmit = canSubmit(it.email, v)
        )
    }

    fun togglePasswordVisibility() = _uiState.update {
        it.copy(passwordVisible = !it.passwordVisible)
    }

    fun onRememberMeChanged(v: Boolean) = _uiState.update { it.copy(rememberMe = v) }

    /* ----------------- Actions ----------------- */
    fun login() = viewModelScope.launch {
        val s = _uiState.value
        if (s.loading) return@launch                    // cegah double tap

        val email = s.email.trim()
        val pass = s.password

        // Validasi cepat di VM (hindari call repo yang sia-sia)
        if (!isValidEmail(email)) {
            _uiState.update { it.copy(errorMessage = "Email tidak valid.") }
            return@launch
        }
        if (pass.length < 6) {
            _uiState.update { it.copy(errorMessage = "Password minimal 6 karakter.") }
            return@launch
        }

        _uiState.update { it.copy(loading = true, errorMessage = null) }

        try {
            // asumsi: repo.login() return Result<Something>
            val result = repo.login(email, pass)
            result.onSuccess {
                // TODO: jika rememberMe true, simpan token/credential ke DataStore
                _uiState.update { it.copy(loading = false, isLoggedIn = true) }
            }.onFailure { e ->
                _uiState.update {
                    it.copy(
                        loading = false,
                        errorMessage = e.message ?: "Login gagal. Coba lagi."
                    )
                }
            }
        } catch (t: Throwable) {
            _uiState.update {
                it.copy(loading = false, errorMessage = "Terjadi masalah jaringan.")
            }
        }
    }

    /* ----------------- Helpers ----------------- */
    // tetap pure (tanpa android.util.Patterns) supaya gampang unit-test
    private val EMAIL_REGEX =
        Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\$")

    private fun isValidEmail(s: String) = EMAIL_REGEX.matches(s)

    private fun canSubmit(email: String, password: String): Boolean =
        isValidEmail(email) && password.length >= 6
}

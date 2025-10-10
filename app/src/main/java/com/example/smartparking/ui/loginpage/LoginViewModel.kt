package com.example.smartparking.ui.loginpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import com.example.smartparking.data.repository.UserRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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

class LoginViewModel (
    private val userRepository: UserRepository = UserRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState: StateFlow<LoginUiState> = _uiState


    fun onEmailChanged(newEmail: String) {
        _uiState.value = _uiState.value.copy(email = newEmail)
    }

    fun onPasswordChanged(newPassword: String) {
        _uiState.value = _uiState.value.copy(password = newPassword)
    }

    fun togglePasswordVisibility() {
        _uiState.value = _uiState.value.copy(
            passwordVisible = !_uiState.value.passwordVisible
        )
    }

    fun onRememberMeChanged(checked: Boolean) {
        _uiState.value = _uiState.value.copy(rememberMe = checked)
    }


    fun login() {
        val email = _uiState.value.email
        val password = _uiState.value.password

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loading = true, errorMessage = null)

            try {
                val response = userRepository.login(email, password)
                if (response.isSuccessful) {
                    val user = response.body()
                    if (user != null) {
                        _uiState.value = _uiState.value.copy(
                            loading = false,
                            isLoggedIn = true,
                            errorMessage = null
                        )
                    } else {
                        _uiState.value = _uiState.value.copy(
                            loading = false,
                            errorMessage = "User not found"
                        )
                    }
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loading = false,
                    errorMessage = e.message
                )
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

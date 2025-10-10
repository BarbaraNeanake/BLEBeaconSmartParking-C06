package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SignUpUiState(
    val name: String = "",
    val email: String = "",
    val licensePlate: String = "",
    val countryCode: String = "+62",
    val phoneNumber: String = "",
    val password: String = "",
    val confirmPassword: String = "",
    val showPassword: Boolean = false,
    val showConfirmPassword: Boolean = false,
    val loading: Boolean = false,
    val error: String? = null,
    val registered: Boolean = false,
    val canSubmit: Boolean = false
) {
    val phoneE164: String get() = countryCode + phoneNumber.filter { it.isDigit() }
}

class SignUpViewModel : ViewModel() {

    private val _ui = MutableStateFlow(SignUpUiState())
    val ui = _ui.asStateFlow()

    /* ----------------- Input handlers ----------------- */
    fun onName(v: String) = _ui.update {
        it.copy(name = v, error = null, canSubmit = canSubmit(v, it.email, it.licensePlate, it.password, it.confirmPassword))
    }

    fun onEmail(v: String) = _ui.update {
        it.copy(email = v.trim(), error = null, canSubmit = canSubmit(it.name, v.trim(), it.licensePlate, it.password, it.confirmPassword))
    }

    fun onLicensePlate(v: String) = _ui.update {
        it.copy(licensePlate = v.uppercase(), error = null, canSubmit = canSubmit(it.name, it.email, v.uppercase(), it.password, it.confirmPassword))
    }

    fun onCountryCode(v: String) = _ui.update { it.copy(countryCode = v, error = null) }

    fun onPhone(v: String) = _ui.update { it.copy(phoneNumber = v, error = null) }

    fun onPassword(v: String) = _ui.update {
        it.copy(password = v, error = null, canSubmit = canSubmit(it.name, it.email, it.licensePlate, v, it.confirmPassword))
    }

    fun onConfirmPassword(v: String) = _ui.update {
        it.copy(confirmPassword = v, error = null, canSubmit = canSubmit(it.name, it.email, it.licensePlate, it.password, v))
    }

    fun togglePwd() = _ui.update { it.copy(showPassword = !it.showPassword) }

    fun toggleConfirmPwd() = _ui.update { it.copy(showConfirmPassword = !it.showConfirmPassword) }

    /* ----------------- Action ----------------- */
    fun register() = viewModelScope.launch {
        val s = _ui.value
        if (s.loading) return@launch

        // Validasi FE
        if (s.name.isBlank()) { fail("Nama wajib diisi"); return@launch }
        if (!isValidEmail(s.email)) { fail("Email tidak valid"); return@launch }
        if (!isValidPlate(s.licensePlate)) { fail("Plat mobil tidak valid"); return@launch }
        if (s.password.length < 6) { fail("Password minimal 6 karakter"); return@launch }
        if (s.password != s.confirmPassword) { fail("Konfirmasi password tidak sama"); return@launch }

        _ui.update { it.copy(loading = true, error = null) }

        // TODO: sambungkan ke repository-mu di sini (simulasi dulu)
        delay(900)

        _ui.update { it.copy(loading = false, registered = true) }
    }

    /* ----------------- Helpers ----------------- */
    private val EMAIL_REGEX = Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\$")
    private fun isValidEmail(s: String) = EMAIL_REGEX.matches(s)

    // Validasi sederhana plat mobil (huruf/angka/spasi/tanda minus), minimal 5 char
    private val PLATE_REGEX = Regex("^[A-Z0-9 -]{5,}\$")
    private fun isValidPlate(s: String) = PLATE_REGEX.matches(s.trim().uppercase())

    private fun canSubmit(name: String, email: String, plate: String, pass: String, confirm: String): Boolean =
        name.isNotBlank() && isValidEmail(email) && isValidPlate(plate) && pass.length >= 6 && pass == confirm

    private fun fail(msg: String) = _ui.update { it.copy(error = msg) }
}

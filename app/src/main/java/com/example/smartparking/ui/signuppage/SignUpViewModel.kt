package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val UI_DATE_FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy")

data class SignUpUiState(
    val name: String = "",
    val email: String = "",
    val licensePlate: String = "",
    val countryCode: String = "+62",
    val phoneNumber: String = "",
    val password: String = "",
    val confirmPassword: String = "",
    val birthDateFormatted: String = "",          // ← ditambahkan agar sinkron dengan UI
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

    /* ----------------- Updaters (1 versi saja) ----------------- */
    fun onName(v: String) = _ui.update {
        val nv = it.copy(name = v, error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    fun onEmail(v: String) = _ui.update {
        val nv = it.copy(email = v.trim(), error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    fun onLicensePlate(v: String) = _ui.update {
        val plate = v.uppercase()
        val nv = it.copy(licensePlate = plate, error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    fun onCountryCode(v: String) = _ui.update {
        it.copy(countryCode = v, error = null)
    }

    fun onPhone(v: String) = _ui.update {
        it.copy(phoneNumber = v, error = null)
    }

    fun onPassword(v: String) = _ui.update {
        val nv = it.copy(password = v, error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    fun onConfirmPassword(v: String) = _ui.update {
        val nv = it.copy(confirmPassword = v, error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    fun togglePwd() = _ui.update { it.copy(showPassword = !it.showPassword) }

    fun toggleConfirmPwd() = _ui.update { it.copy(showConfirmPassword = !it.showConfirmPassword) }

    /** Terima millis dari DatePicker, simpan sebagai string dd/MM/yyyy. */
    fun onBirthDateMillis(millis: Long) = _ui.update {
        val dateStr = Instant.ofEpochMilli(millis)
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
            .format(UI_DATE_FMT)
        val nv = it.copy(birthDateFormatted = dateStr, error = null)
        nv.copy(canSubmit = canSubmit(nv))
    }

    /* ----------------- Action ----------------- */
    fun register() = viewModelScope.launch {
        val s = _ui.value
        if (s.loading) return@launch

        // Validasi FE
        if (s.name.isBlank()) { fail("Nama wajib diisi"); return@launch }
        if (!isValidEmail(s.email)) { fail("Email tidak valid"); return@launch }
        if (!isValidPlate(s.licensePlate)) { fail("Plat kendaraan tidak valid"); return@launch }
        if (s.birthDateFormatted.isBlank()) { fail("Tanggal lahir wajib diisi"); return@launch }
        if (s.phoneNumber.isBlank()) { fail("Nomor HP wajib diisi"); return@launch }
        if (s.password.length < 6) { fail("Password minimal 6 karakter"); return@launch }
        if (s.password != s.confirmPassword) { fail("Konfirmasi password tidak sama"); return@launch }

        // Cek format tanggal
        try {
            LocalDate.parse(s.birthDateFormatted, UI_DATE_FMT)
        } catch (_: Exception) {
            fail("Format tanggal lahir tidak valid (gunakan dd/MM/yyyy)")
            return@launch
        }

        _ui.update { it.copy(loading = true, error = null) }

        // Simulasi call network / repository
        delay(900)

        _ui.update { it.copy(loading = false, registered = true) }
    }

    /* ----------------- Helpers ----------------- */
    private val EMAIL_REGEX = Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\$")
    private fun isValidEmail(s: String) = EMAIL_REGEX.matches(s)

    // Validasi sederhana plat kendaraan
    private val PLATE_REGEX = Regex("^[A-Z0-9 -]{5,}\$")
    private fun isValidPlate(s: String) = PLATE_REGEX.matches(s.trim().uppercase())

    private fun canSubmit(s: SignUpUiState): Boolean =
        s.name.isNotBlank() &&
                isValidEmail(s.email) &&
                isValidPlate(s.licensePlate) &&
                s.birthDateFormatted.isNotBlank() &&
                s.phoneNumber.isNotBlank() &&
                s.password.length >= 6 &&
                s.password == s.confirmPassword

    private fun fail(msg: String) = _ui.update { it.copy(error = msg, loading = false) }
}

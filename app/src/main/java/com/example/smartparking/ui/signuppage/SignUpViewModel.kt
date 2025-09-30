package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

data class SignUpUiState(
    val name: String = "",
    val email: String = "",
    val birthDateMillis: Long? = null, // disimpan millis, render sebagai dd/MM/yyyy
    val countryCode: String = "+62",
    val phoneNumber: String = "",
    val password: String = "",
    val showPassword: Boolean = false,
    val loading: Boolean = false,
    val error: String? = null,
    val registered: Boolean = false
) {
    val birthDateFormatted: String
        get() = birthDateMillis?.let {
            val d = Instant.ofEpochMilli(it).atZone(ZoneId.systemDefault()).toLocalDate()
            d.format(DateTimeFormatter.ofPattern("dd/MM/yyyy"))
        } ?: ""
    val phoneE164: String get() = countryCode + phoneNumber.filter { it.isDigit() }
}

class SignUpViewModel(
    private val repo: AuthRepository = FakeAuthRepository()
) : ViewModel() {

    private val _ui = MutableStateFlow(SignUpUiState())
    val ui = _ui.asStateFlow()

    fun onName(v: String) { _ui.value = _ui.value.copy(name = v) }
    fun onEmail(v: String) { _ui.value = _ui.value.copy(email = v) }
    fun onBirthDateMillis(v: Long) { _ui.value = _ui.value.copy(birthDateMillis = v) }
    fun onCountryCode(v: String) { _ui.value = _ui.value.copy(countryCode = v) }
    fun onPhone(v: String) { _ui.value = _ui.value.copy(phoneNumber = v) }
    fun onPassword(v: String) { _ui.value = _ui.value.copy(password = v) }
    fun togglePwd() { _ui.value = _ui.value.copy(showPassword = !_ui.value.showPassword) }

    fun register() = viewModelScope.launch {
        val s = _ui.value
        // validasi ringan di FE
        if (s.name.isBlank()) { fail("Nama wajib diisi"); return@launch }
        if (!s.email.contains("@")) { fail("Email tidak valid"); return@launch }
        if (s.birthDateMillis == null) { fail("Tanggal lahir wajib diisi"); return@launch }
        if (s.phoneNumber.isBlank()) { fail("Nomor HP wajib diisi"); return@launch }
        if (s.password.length < 6) { fail("Password minimal 6 karakter"); return@launch }

        _ui.value = s.copy(loading = true, error = null)

        val birthIso = Instant.ofEpochMilli(s.birthDateMillis)
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
            .format(DateTimeFormatter.ISO_LOCAL_DATE) // "yyyy-MM-dd"

        val r = repo.register(
            name = s.name.trim(),
            email = s.email.trim(),
            birthDateIso = birthIso,
            phoneE164 = s.phoneE164,
            password = s.password
        )
        _ui.value = _ui.value.copy(loading = false)
        r.onSuccess { _ui.value = _ui.value.copy(registered = true) }
            .onFailure { fail(it.message ?: "Register gagal") }
    }

    private fun fail(msg: String) { _ui.value = _ui.value.copy(error = msg) }
}

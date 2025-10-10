package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import com.example.smartparking.data.model.User
import com.example.smartparking.data.repository.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val UI_DATE_FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy")

data class SignUpUiState(
    val name: String = "",
    val email: String = "",
    val birthDate: String = "",
    val countryCode: String = "+62",
    val phoneNumber: String = "",
    val password: String = "",
    val showPassword: Boolean = false,
    val loading: Boolean = false,
    val error: String? = null,
    val registered: Boolean = false
) {
    val birthDateFormatted: String get() = birthDate
    val phoneE164: String get() = countryCode + phoneNumber.filter { it.isDigit() }
}

class SignUpViewModel(
    private val userRepository: UserRepository = UserRepository()
) : ViewModel() {

    private val _ui = MutableStateFlow(SignUpUiState())
    val ui = _ui.asStateFlow()

    fun onName(v: String) { _ui.value = _ui.value.copy(name = v) }
    fun onEmail(v: String) { _ui.value = _ui.value.copy(email = v) }
    fun onBirthDateMillis(millis: Long) {
        val dateStr = Instant.ofEpochMilli(millis)
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
            .format(UI_DATE_FMT)
        _ui.value = _ui.value.copy(birthDate = dateStr)
    }

    fun onCountryCode(v: String) { _ui.value = _ui.value.copy(countryCode = v) }
    fun onPhone(v: String) { _ui.value = _ui.value.copy(phoneNumber = v) }
    fun onPassword(v: String) { _ui.value = _ui.value.copy(password = v) }
    fun togglePwd() { _ui.value = _ui.value.copy(showPassword = !_ui.value.showPassword) }

    fun register() = viewModelScope.launch {
        val s = _ui.value
        // basic FE validation
        if (s.name.isBlank()) { fail("Nama wajib diisi"); return@launch }
        if (!s.email.contains("@")) { fail("Email tidak valid"); return@launch }
        if (s.birthDate.isBlank()) { fail("Tanggal lahir wajib diisi"); return@launch }
        if (s.phoneNumber.isBlank()) { fail("Nomor HP wajib diisi"); return@launch }
        if (s.password.length < 6) { fail("Password minimal 6 karakter"); return@launch }

        _ui.value = s.copy(loading = true, error = null)

        val birthIso = try {
            LocalDate.parse(s.birthDate, UI_DATE_FMT)
                .format(DateTimeFormatter.ISO_LOCAL_DATE)
        } catch (e: Exception) {
            _ui.value = s.copy(loading = false)
            fail("Format tanggal lahir tidak valid (gunakan dd/MM/yyyy)")
            return@launch
        }

        val req = User(
            nama = s.name,
            email = s.email,
            license = s.phoneNumber,
            password = s.password,
            birthdate = s.birthDate,
            roles = "Mahasiswa"
        )

        val r = userRepository.createUser(req)

        if(r.isSuccessful){
            val user = r.body()
            if (user != null) {
                _ui.value = _ui.value.copy(
                    loading = false,
                    registered = true
                )
            } else {
                _ui.value = _ui.value.copy(
                    loading = false,
                )
            }
        }
    }

    private fun fail(msg: String) { _ui.value = _ui.value.copy(error = msg) }
}

package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.auth.AuthRepository
import com.example.smartparking.data.auth.FakeAuthRepository
import com.example.smartparking.data.model.User
import com.example.smartparking.data.repository.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

private val UI_DATE_FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy")

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

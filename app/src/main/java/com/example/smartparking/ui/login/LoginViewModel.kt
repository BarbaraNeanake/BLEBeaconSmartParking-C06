package com.example.smartparking.ui.login

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class LoginViewModel : ViewModel() {
    private val _loginState = MutableStateFlow<Boolean?>(null)
    val loginState: StateFlow<Boolean?> = _loginState

    fun login(email: String, password: String) {
        // TODO: Replace this with real DB validation
        _loginState.value = (email == "test@example.com" && password == "1234")
    }
}

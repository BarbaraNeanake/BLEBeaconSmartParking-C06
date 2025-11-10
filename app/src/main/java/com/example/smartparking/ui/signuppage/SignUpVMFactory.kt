package com.example.smartparking.ui.signuppage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.smartparking.data.repository.UserRepository

class SignUpVMFactory(
    private val repo: UserRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(SignUpViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return SignUpViewModel(repo) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}

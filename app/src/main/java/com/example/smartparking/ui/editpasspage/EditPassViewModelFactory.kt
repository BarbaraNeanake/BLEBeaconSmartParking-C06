package com.example.smartparking.ui.editpasspage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.smartparking.data.repository.UserRepository

class EditPassVMFactory(
    private val repo: UserRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(EditPassViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return EditPassViewModel(repo) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}

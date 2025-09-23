package com.example.smartparking.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.User
import com.example.smartparking.data.remote.RetrofitClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class HomeViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState

    private val apiService = RetrofitClient.instance

    fun setUsername(name: String) {
        _uiState.value = _uiState.value.copy(nama = name)
    }

    fun loadUserById(id: Int) {
        viewModelScope.launch {
            try {
                println("Memanggil API untuk userId=$id")  // debug
                val response = apiService.getUserById(id)
                println("Response: $response")  // debug
                val user = response.body()
                user?.let {
                    _uiState.value = _uiState.value.copy(nama = it.nama)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
}

package com.example.smartparking.ui.homepage

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// status parkir yg ditampilkan di InfoCard
data class ParkingStatus(
    val totalSlots: Int = 147,
    val usedSlots: Int = 65
)

class HomePageViewModel : ViewModel() {

    private val _parkingStatus = mutableStateOf(ParkingStatus())
    val parkingStatus: State<ParkingStatus> get() = _parkingStatus

    // ⬇️ ini baru: khusus miniatur 5 slot
    private val _miniatureStatus = mutableStateOf(ParkingStatus(totalSlots = 5, usedSlots = 0))
    val miniatureStatus: State<ParkingStatus> get() = _miniatureStatus

    fun fetchParkingStatus() {
        viewModelScope.launch {
            // Simulasi fetch (biar terlihat smooth di UI); ganti dengan API nanti
            delay(0)
            _parkingStatus.value = ParkingStatus(totalSlots = 147, usedSlots = 65)
        }
    }

    // ⬇️ ini baru: tinggal temen BE isi dari API miniatur
    fun fetchMiniatureStatus() {
        viewModelScope.launch {
            // ganti nanti pakai response BE
            delay(0)
            _miniatureStatus.value = ParkingStatus(totalSlots = 5, usedSlots = 0)
        }
    }
}


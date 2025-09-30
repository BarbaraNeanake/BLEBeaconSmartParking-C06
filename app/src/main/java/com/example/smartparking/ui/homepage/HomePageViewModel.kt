package com.example.smartparking.ui.homepage

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

// status parkir yg ditampilkan di InfoCard
data class ParkingStatus(
    val totalSlots: Int = 147,
    val usedSlots: Int = 65
)

class HomePageViewModel : ViewModel() {

    private val _parkingStatus = mutableStateOf(ParkingStatus())
    val parkingStatus: State<ParkingStatus> get() = _parkingStatus

    fun fetchParkingStatus() {
        viewModelScope.launch {
            // TODO: ganti dengan call ke API nanti
            _parkingStatus.value = ParkingStatus(totalSlots = 147, usedSlots = 65)
        }
    }
}

package com.example.smartparking.ui.homepage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.Parking
import com.example.smartparking.data.remote.ParkingStats
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.ParkingRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ParkingStatus(
    val totalSlots: Int = 147,
    val usedSlots: Int = 65
)

class HomePageViewModel(
    private val repo: ParkingRepository = ParkingRepository(RetrofitProvider.parkingApi)
) : ViewModel() {

    private val _parkingStatus = MutableStateFlow(ParkingStatus())
    val parkingStatus: StateFlow<ParkingStatus> = _parkingStatus

    private val _miniatureStatus = MutableStateFlow(ParkingStatus())
    val miniatureStatus: StateFlow<ParkingStatus> = _miniatureStatus

    private val _stats = MutableStateFlow<ParkingStats?>(null)
    val stats = _stats.asStateFlow()

    fun loadStats() {
        viewModelScope.launch {
            try {
                _stats.value = repo.getStats()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun fetchParkingStatus() {
        viewModelScope.launch {
            _parkingStatus.value = ParkingStatus(totalSlots = 147, usedSlots = 65)
        }
    }

    fun fetchMiniatureStatus() {
        viewModelScope.launch {
            try {
                val resp = repo.getParkings()
                if (resp.isSuccessful) {
                    val data: List<Parking> = resp.body().orEmpty()
                    val total = data.size
                    val used = data.count { it.status.equals("occupied", ignoreCase = true) }
                    _parkingStatus.value = ParkingStatus(total, used)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
}

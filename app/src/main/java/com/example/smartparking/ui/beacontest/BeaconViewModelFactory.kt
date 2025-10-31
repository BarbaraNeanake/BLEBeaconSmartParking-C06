//package com.example.smartparking.ui.beacontest
//
//import androidx.lifecycle.ViewModel
//import androidx.lifecycle.ViewModelProvider
//import com.example.smartparking.data.repository.ParkingRepository
//
//class BeaconViewModelFactory(
//    private val repository: ParkingRepository
//) : ViewModelProvider.Factory {
//    override fun <T : ViewModel> create(modelClass: Class<T>): T {
//        if (modelClass.isAssignableFrom(BeaconViewModel::class.java)) {
//            @Suppress("UNCHECKED_CAST")
//            return BeaconViewModel(repository) as T
//        }
//        throw IllegalArgumentException("Unknown ViewModel class")
//    }
//}

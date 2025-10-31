// LiveParkingVMFactory.kt (taruh di paket yang sama dengan ViewModel)
package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.ParkingRepository
import com.example.smartparking.data.repository.dao.SessionDao

class LiveParkingVMFactory(
    private val sessionDao: SessionDao
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(LiveParkingViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return LiveParkingViewModel(sessionDao = sessionDao) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}


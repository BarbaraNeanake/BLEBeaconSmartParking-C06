package com.example.smartparking.data.repository

import com.example.smartparking.data.remote.ParkingApiService
import com.example.smartparking.data.model.Parking
import retrofit2.Response

class ParkingRepository(
    private val apiService: ParkingApiService
) {
    // GET /parking
    suspend fun getParkings(): Response<List<Parking>> {
        return apiService.getParkings()
    }

    // GET /parking/:id
    suspend fun getParkingById(id: Int): Response<Parking> {
        return apiService.getParkingById(id)
    }

    // POST /parking
    suspend fun addParking(parking: Parking): Response<Parking> {
        return apiService.addParking(parking)
    }

    // DELETE /parking/:id
    suspend fun removeParking(id: Int): Response<Unit> {
        return apiService.removeParking(id)
    }

    // PATCH /parking/:nomor
    suspend fun updateParking(nomor: Int, parking: Parking): Response<Parking> {
        return apiService.updateParking(nomor, parking)
    }
}

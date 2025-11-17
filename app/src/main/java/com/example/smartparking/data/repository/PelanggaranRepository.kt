package com.example.smartparking.data.repository

import com.example.smartparking.data.remote.PelanggaranApiService
import com.example.smartparking.data.model.Pelanggaran
import retrofit2.Response

class PelanggaranRepository(
    private val api: PelanggaranApiService
) {
    suspend fun getAll(): Response<List<Pelanggaran>> {
        return api.getAll()
    }

    suspend fun getByUser(userId: Int?): Response<List<Pelanggaran>> {
        return api.getByUser(userId)
    }
}

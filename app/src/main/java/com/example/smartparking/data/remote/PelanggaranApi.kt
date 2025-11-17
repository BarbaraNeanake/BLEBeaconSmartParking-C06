package com.example.smartparking.data.remote

import com.example.smartparking.data.model.Pelanggaran
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Path

interface PelanggaranApiService {
    // ambil semua pelanggaran (opsional)
    @GET("pelanggaran")
    suspend fun getAll(): Response<List<Pelanggaran>>

    @GET("pelanggaran/{userId}")
    suspend fun getByUser(@Path("userId") userId: Int?): Response<List<Pelanggaran>>
}

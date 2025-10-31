package com.example.smartparking.data.remote

import com.example.smartparking.data.model.Parking
import retrofit2.Response
import retrofit2.http.*

data class UpdateParkingRequest(
    val status: String,      // "occupied" | "available" | "disabled_slot"
    val userid: Int? = null,  // null saat free
    val rolesUser: String? = null
)

interface ParkingApiService {

    // GET /parking/
    @GET("parking")
    suspend fun getParkings(): Response<List<Parking>>

    // GET /parking/:id
    @GET("parking/{id}")
    suspend fun getParkingById(
        @Path("id") id: Int
    ): Response<Parking>

    // POST /parking/
    @POST("parking")
    suspend fun addParking(
        @Body request: Parking
    ): Response<Parking>

    // DELETE /parking/:id
    @DELETE("parking/{id}")
    suspend fun removeParking(
        @Path("id") id: Int
    ): Response<Unit>

    // PATCH /parking/:nomor
    @PATCH("parking/{nomor}")
    suspend fun updateParking(
        @Path("nomor") nomor: Int,
        @Body request: UpdateParkingRequest
    ): Response<Parking>
}

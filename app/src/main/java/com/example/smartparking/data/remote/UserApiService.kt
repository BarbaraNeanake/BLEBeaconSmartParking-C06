package com.example.smartparking.data.remote

import com.example.smartparking.data.model.LoginResponse
import com.example.smartparking.data.model.User
import retrofit2.Response
import retrofit2.http.*

// Request body untuk login
data class LoginRequest(
    val email: String,
    val password: String
)

// Response body untuk login
data class LoginResponse(
    val message: String,
    val token: String,     // ✅ backend sekarang kirim token
    val user: User
)

interface UserApiService {

    // 🔹 Get all users
    @GET("users")
    suspend fun getUsers(): List<User>

    // 🔹 Create new user
    @POST("users")
    suspend fun createUser(@Body user: User): Response<User>

    // 🔹 Delete user
    @DELETE("users/{id}")
    suspend fun deleteUser(@Path("id") userId: Int): Response<Unit>

    // 🔹 Get user by ID
    @GET("users/{id}")
    suspend fun getUserById(@Path("id") id: Int): Response<User>

    // 🔹 Login user (mengembalikan token + data user)
    @POST("users/login")
    suspend fun login(@Body request: LoginRequest): Response<LoginResponse>
}

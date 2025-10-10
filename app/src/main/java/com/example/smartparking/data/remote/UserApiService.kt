package com.example.smartparking.data.remote

import android.R
import com.example.smartparking.data.model.User
import retrofit2.Response
import retrofit2.http.*

data class LoginRequest(
    val email: String,
    val password: String
)



interface UserApiService {

    @GET("users")
    suspend fun getUsers(): List<User>

    @POST("users")
    suspend fun createUser(@Body user: User): Response<User>

    @DELETE("users/{id}")
    suspend fun deleteUser(@Path("id") userId: Int): Response<Unit>

    @GET("users/{id}")
    suspend fun getUserById(@Path("id") id: Int): Response<User>

    @POST("users/login")
    suspend fun login(@Body request: LoginRequest): Response<User>
}

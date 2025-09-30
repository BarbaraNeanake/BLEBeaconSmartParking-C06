package com.example.smartparking.data.remote

import android.R
import com.example.smartparking.data.model.User
import retrofit2.Response
import retrofit2.http.*

interface UserApiService {

    @GET("users")
    suspend fun getUsers(): List<User>

    @POST("users")
    suspend fun createUser(@Body user: R.string): Response<User>

    @DELETE("users/{id}")
    suspend fun deleteUser(@Path("id") userId: Int): Response<Unit>

    @GET("users/{id}")
    suspend fun getUserById(@Path("id") id: Int): Response<User>
}

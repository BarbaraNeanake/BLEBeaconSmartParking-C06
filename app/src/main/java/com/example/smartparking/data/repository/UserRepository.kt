package com.example.smartparking.data.repository

import android.R
import com.example.smartparking.data.model.User
import com.example.smartparking.data.remote.LoginRequest
import com.example.smartparking.data.remote.RetrofitClient
import retrofit2.Response

class UserRepository {
    suspend fun getUsers(): List<User> {
        return RetrofitClient.userApi.getUsers()
    }

    suspend fun createUser(user: User): Response<User> {
        return RetrofitClient.userApi.createUser(user)
    }

    suspend fun deleteUser(id: Int): Response<Unit> {
        return RetrofitClient.userApi.deleteUser(id)
    }

    suspend fun getUserById(id:Int): Response<User> {
        return RetrofitClient.userApi.getUserById(id)
    }

    suspend fun login(email: String, password: String): Response<User> {
        val request = LoginRequest(email, password)
        return RetrofitClient.userApi.login(request)
    }


}

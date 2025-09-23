package com.example.smartparking.data.repository

import android.R
import com.example.smartparking.data.model.User
import com.example.smartparking.data.remote.RetrofitClient
import retrofit2.Response

class UserRepository {
    suspend fun getUsers(): List<User> {
        return RetrofitClient.instance.getUsers()
    }

    suspend fun createUser(user: R.string): Response<User> {
        return RetrofitClient.instance.createUser(user)
    }

    suspend fun deleteUser(id: Int): Response<Unit> {
        return RetrofitClient.instance.deleteUser(id)
    }

    suspend fun getUserById(id:Int): Response<User> {
        return RetrofitClient.instance.getUserById(id)
    }

}

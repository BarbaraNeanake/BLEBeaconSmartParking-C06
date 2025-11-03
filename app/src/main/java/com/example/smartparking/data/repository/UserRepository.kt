package com.example.smartparking.data.repository

import android.R
import com.example.smartparking.data.model.SessionEntity
import com.example.smartparking.data.model.User
import com.example.smartparking.data.remote.LoginRequest
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.remote.UpdatePasswordRequest
import com.example.smartparking.data.repository.dao.SessionDao
import retrofit2.Response

class UserRepository(
    private val sessionDao: SessionDao    // ✅ tambahkan ini
) {
    suspend fun getUsers(): List<User> =
        RetrofitProvider.userApi.getUsers()

    suspend fun createUser(user: User): Response<User> =
        RetrofitProvider.userApi.createUser(user)

    suspend fun deleteUser(id: Int): Response<Unit> =
        RetrofitProvider.userApi.deleteUser(id)

    suspend fun getUserById(id: Int): Response<User> =
        RetrofitProvider.userApi.getUserById(id)

    suspend fun login(email: String, password: String): Result<Unit> = runCatching {
        val response = RetrofitProvider.userApi.login(LoginRequest(email, password))
        if (!response.isSuccessful || response.body() == null) {
            throw Exception("Login gagal: ${response.code()} ${response.message()}")
        }

        val resp = response.body()!!
        val u = resp.user

        val session = SessionEntity(
            token = resp.token,
            userId = u.userID,
            name = u.nama,
            email = u.email,
            role = u.roles
        )

        sessionDao.upsert(session)   // ✅ sekarang pakai instance
    }

    suspend fun updatePassword(email: String, newPassword: String): Result<String> = runCatching {
        val resp = RetrofitProvider.userApi.updatePassword(
            UpdatePasswordRequest(email = email, password = newPassword)
        )
        if (!resp.isSuccessful) {
            throw Exception("Gagal update password: HTTP ${resp.code()} ${resp.message()}")
        }
        resp.body()?.message ?: "Password updated"
    }

    suspend fun logout() {
        sessionDao.clear()
    }
}

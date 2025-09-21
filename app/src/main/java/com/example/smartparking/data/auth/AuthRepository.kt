package com.example.smartparking.data.auth

import kotlinx.coroutines.delay

data class AuthedUser(val email: String)

/** Kontrak ke backend */
interface AuthRepository {
    suspend fun login(email: String, password: String): Result<AuthedUser>
}

/** Implementasi sementara agar app jalan tanpa BE */
class FakeAuthRepository : AuthRepository {
    override suspend fun login(email: String, password: String): Result<AuthedUser> {
        delay(700) // simulasi network
        return if (email.endsWith("@ugm.ac.id") && password.length >= 6) {
            Result.success(AuthedUser(email))
        } else {
            Result.failure(IllegalArgumentException("Email harus @ugm.ac.id & password ≥ 6 karakter"))
        }
    }
}

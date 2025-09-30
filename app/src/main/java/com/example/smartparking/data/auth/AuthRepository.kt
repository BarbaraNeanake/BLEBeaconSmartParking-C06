package com.example.smartparking.data.auth

import kotlinx.coroutines.delay

data class AuthedUser(
    val id: String = "temp-id",
    val name: String,
    val email: String
)

/** Kontrak ke backend (nantinya kamu ganti implementasi ke service NeonDB) */
interface AuthRepository {
    suspend fun login(email: String, password: String): Result<AuthedUser>
    suspend fun register(
        name: String,
        email: String,
        birthDateIso: String,   // "2005-04-23" (ISO)
        phoneE164: String,      // contoh: +628123456789
        password: String
    ): Result<AuthedUser>
}

/** Dummy agar app bisa jalan tanpa BE */
class FakeAuthRepository : AuthRepository {
    override suspend fun login(email: String, password: String): Result<AuthedUser> {
        delay(600)
        return if (email.contains("@") && password.length >= 6)
            Result.success(AuthedUser(name = "User", email = email))
        else Result.failure(IllegalArgumentException("Email/Password tidak valid"))
    }

    override suspend fun register(
        name: String,
        email: String,
        birthDateIso: String,
        phoneE164: String,
        password: String
    ): Result<AuthedUser> {
        delay(900)
        if (name.isBlank()) return Result.failure(IllegalArgumentException("Nama wajib diisi"))
        if (!email.contains("@")) return Result.failure(IllegalArgumentException("Email tidak valid"))
        if (!phoneE164.startsWith("+")) return Result.failure(IllegalArgumentException("Kode negara wajib, contoh +62"))
        if (password.length < 6) return Result.failure(IllegalArgumentException("Password ≥ 6 karakter"))
        // sukses
        return Result.success(AuthedUser(name = name, email = email))
    }
}

package com.example.smartparking.data.session

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

class SecureSession(context: Context) {

    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val prefs = EncryptedSharedPreferences.create(
        context,
        "secure_session",
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )

    fun save(token: String, userId: Int) {
        prefs.edit()
            .putString("token", token)
            .putInt("userId", userId)
            .apply()
    }

    fun read(): Pair<String, Int>? {
        val t = prefs.getString("token", null) ?: return null
        val id = prefs.getInt("userId", -1).takeIf { it >= 0 } ?: return null
        return t to id
    }

    fun clear() = prefs.edit().clear().apply()
}

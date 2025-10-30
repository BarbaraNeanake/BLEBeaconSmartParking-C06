package com.example.smartparking.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "session")
data class SessionEntity(
    @PrimaryKey val id: Int = 0,          // single-row table
    val token: String,
    val userId: Int?,
    val name: String,
    val email: String,
    val role: String,
    val loggedAt: Long = System.currentTimeMillis()
)

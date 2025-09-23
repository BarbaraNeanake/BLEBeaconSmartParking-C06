package com.example.smartparking.data.model

import java.time.LocalDateTime

data class LogActivity(
    val numActivity: Int,
    val id: Int?,
    val numPark: Int?,
    val time: LocalDateTime,
    val status: String
)

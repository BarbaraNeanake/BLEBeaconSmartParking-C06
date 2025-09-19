package com.example.smartparking.data.model

data class Parking(
    val numPark: Int,
    val userID: Int?,         // bisa null kalau tidak ada user yang parkir
    val loc: String,
    val status: String
)
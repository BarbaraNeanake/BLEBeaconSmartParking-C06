package com.example.smartparking.data.model

data class Parking(
    val nomor: Int,
    val userid: Int?,
    val lokasi: String,
    val status: String,
    val rolesUser: String
)
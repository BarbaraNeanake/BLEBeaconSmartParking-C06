package com.example.smartparking.data.model

data class Parking(
    val nomor: Int? = null,
    val userid: Int? = null,
    val lokasi: String? = null,      // area
    val slot_id: String? = null,     // ID slot unik
    val status: String? = null,
    val rolesUser: String? = null
)

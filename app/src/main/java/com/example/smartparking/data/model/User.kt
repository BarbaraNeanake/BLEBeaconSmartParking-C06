package com.example.smartparking.data.model

data class User(
    val userID: Int? = null,
    val nama: String,
    val roles: String,
    val license: String,
    val email: String,
    val birthdate: String,
    val password: String
)

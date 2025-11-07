package com.example.smartparking.data.model

import com.google.gson.annotations.SerializedName

data class User(
    @SerializedName("userid")
    val userId: Int? = null,

    @SerializedName("nama")
    val nama: String,

    @SerializedName("roles")
    val roles: String,

    @SerializedName("license")
    val license: String? = null,

    @SerializedName("email")
    val email: String,

    @SerializedName("password")
    val password: String? = null,

    @SerializedName("status_user")
    val status: String? = null
)

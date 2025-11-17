package com.example.smartparking.data.model

import com.google.gson.annotations.SerializedName

data class LogActivity(
    @SerializedName("num_activity")
    val numActivity: Int? = null,

    @SerializedName("id")
    val id: Int? = null,

    @SerializedName("nomor")
    val numPark: String? = null,

    @SerializedName("time")
    val time: String? = null,

    @SerializedName("status")
    val status: String? = null,

    @SerializedName("area")
    val area: String? = null
)

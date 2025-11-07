package com.example.smartparking.data.remote

import com.example.smartparking.data.model.LogActivity
import retrofit2.Response
import retrofit2.http.*

data class LogEventRequest(
    val userid: Int,
    val area: String,
    val status: String,
)

interface LogActivityApiService {

    @GET("logs")
    suspend fun getAllLogs(): Response<List<LogActivity>>

    @GET("logs/{id}")
    suspend fun getLogById(@Path("id") id: Int): Response<List<LogActivity>>

    @DELETE("logs/{id}")
    suspend fun deleteLog(@Path("id") id: Int): Response<LogActivity>

    @POST("logs/gate/in")
    suspend fun logGateIn(@Body body: Map<String, Any>): Response<LogActivity>

    @POST("logs/gate/out")
    suspend fun logGateOut(@Body body: Map<String, Any>): Response<LogActivity>

    @POST("logs/event")
    suspend fun sendLogEvent(@Body body: LogEventRequest): Response<LogActivity>}

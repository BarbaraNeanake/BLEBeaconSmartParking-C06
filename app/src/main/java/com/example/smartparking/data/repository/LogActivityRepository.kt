package com.example.smartparking.data.repository

import com.example.smartparking.data.remote.LogActivityApiService
import retrofit2.Response
import com.example.smartparking.data.model.LogActivity
import com.example.smartparking.data.remote.LogEventRequest

class LogActivityRepository(
    private val apiService : LogActivityApiService
) {
    suspend fun getLogs(): Response<List<LogActivity>> {
        return apiService.getAllLogs()
    }

    suspend fun getLogById(userid: Int): Response<List<LogActivity>> {
        return apiService.getLogById(userid)
    }

    suspend fun sendLog(userid: Int, area: String, status: String): Response<LogActivity> {
        return apiService.sendLogEvent(LogEventRequest(userid, area, status))
    }


}

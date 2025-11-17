package com.example.smartparking.data.remote

import com.example.smartparking.data.network.AuthInterceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import com.example.smartparking.BuildConfig
import com.example.smartparking.data.remote.PelanggaranApiService

object RetrofitProvider {

    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor(AuthInterceptor())
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .build()
    }

    val retrofit = Retrofit.Builder()
        .baseUrl(BuildConfig.BASE_URL)
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val userApi: UserApiService by lazy { retrofit.create(UserApiService::class.java) }
    val parkingApi: ParkingApiService by lazy { retrofit.create(ParkingApiService::class.java) }
    val logActivityApi: LogActivityApiService by lazy { retrofit.create(LogActivityApiService::class.java) }
    val pelanggaranApi: PelanggaranApiService = retrofit.create(PelanggaranApiService::class.java)
}

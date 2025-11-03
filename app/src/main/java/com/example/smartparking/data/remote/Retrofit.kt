package com.example.smartparking.data.remote

import com.example.smartparking.data.network.AuthInterceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitProvider {

    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor(AuthInterceptor())
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .build()
    }

    val retrofit = Retrofit.Builder()
        .baseUrl("http://172.20.80.1:3000/") // ganti sesuai server kamu
        .client(client)
        .addConverterFactory(GsonConverterFactory.create()) // ✅ pakai GSON
        .build()

    val userApi: UserApiService by lazy { retrofit.create(UserApiService::class.java) }
    val parkingApi: ParkingApiService by lazy { retrofit.create(ParkingApiService::class.java) }
}

package com.example.smartparking.data.network

import okhttp3.Interceptor
import okhttp3.Response

class AuthInterceptor : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        val original = chain.request()
        val urlPath = original.url.encodedPath

        if (urlPath.contains("/auth/login") || urlPath.contains("/auth/refresh")) {
            return chain.proceed(original)
        }

        val token = TokenProvider.getToken()

        if (token.isNullOrBlank()) return chain.proceed(original)

        val authed = original.newBuilder()
            .header("Authorization", "Bearer $token")
            .build()

        return chain.proceed(authed)
    }
}

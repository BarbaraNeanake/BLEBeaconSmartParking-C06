package com.example.smartparking.data.network

import com.example.smartparking.data.repository.dao.SessionDao
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicReference

object TokenProvider {
    private val cachedToken = AtomicReference<String?>(null)

    fun init(sessionDao: SessionDao) {
        CoroutineScope(Dispatchers.IO).launch {
            cachedToken.set(sessionDao.getSessionOnce()?.token)
            sessionDao.observeSession()
                .onEach { sess -> cachedToken.set(sess?.token) }
                .filterNotNull()
                .collect { /* no-op: hanya menjaga sinkron */ }
        }
    }

    fun getToken(): String? = cachedToken.get()

    // Opsional kalau mau reset manual saat logout tanpa menunggu Flow
    fun reset() { cachedToken.set(null) }
}

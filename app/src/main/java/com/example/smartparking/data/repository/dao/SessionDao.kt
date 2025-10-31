package com.example.smartparking.data.repository.dao

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert   // ✅ pastikan import ini
import com.example.smartparking.data.model.SessionEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface SessionDao {

    @Query("SELECT * FROM session WHERE id = 0")
    fun observeSession(): Flow<SessionEntity?>

    @Query("SELECT * FROM session WHERE id = 0")
    suspend fun getSessionOnce(): SessionEntity?

    @Upsert
    suspend fun upsert(entity: SessionEntity)   // ✅ Room 2.5+ mendukung

    @Query("DELETE FROM session")
    suspend fun clear()
}

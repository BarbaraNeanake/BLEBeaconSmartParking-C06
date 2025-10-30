package com.example.smartparking.data.repository.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.smartparking.data.model.SessionEntity
import com.example.smartparking.data.repository.dao.SessionDao

@Database(
    entities = [SessionEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun sessionDao(): SessionDao
}

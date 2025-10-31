// data/repository/db/AppDatabase.kt
package com.example.smartparking.data.repository.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
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

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            // 1) Ambil cache kalau ada
            val cached = INSTANCE
            if (cached != null) return cached

            // 2) Buat baru secara thread-safe
            return synchronized(this) {
                val again = INSTANCE
                if (again != null) {
                    again
                } else {
                    val created = Room.databaseBuilder(
                        context.applicationContext,
                        AppDatabase::class.java,
                        "smartparking.db"
                    )
                        // .fallbackToDestructiveMigration() // opsional
                        .build()
                    INSTANCE = created
                    created
                }
            }
        }
    }
}

package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.repository.LogActivityRepository
import com.example.smartparking.data.repository.ParkingRepository
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.dao.SessionDao
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class LiveParkingViewModel(
    private val repo: ParkingRepository = ParkingRepository(RetrofitProvider.parkingApi),
    private val sessionDao: SessionDao,
    private val logRepo: LogActivityRepository = LogActivityRepository(RetrofitProvider.logActivityApi)
) : ViewModel() {

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    private val _statusById = MutableStateFlow<Map<String, String>>(emptyMap())
    val statusById: StateFlow<Map<String, String>> = _statusById

    private var cacheSlotToNomor: Map<String, Int> = emptyMap()

    // SLOT YANG SEDANG DITEMPATI USER SAAT INI
    private var lastOccupiedSlotId: String? = null

    init {
        reload()
    }

    fun reload() {
        if (_loading.value) return
        viewModelScope.launch {
            _loading.value = true
            _error.value = null
            try {
                val resp = repo.getParkings()
                if (!resp.isSuccessful) throw Exception("HTTP ${resp.code()} ${resp.message()}")

                val rows = resp.body().orEmpty()

                val statusMap = rows.associate { row ->
                    val id = extractSlotId(row.lokasi) ?: "S${row.nomor}"
                    id to (row.status?.lowercase() ?: "available")
                }
                _statusById.value = statusMap

                cacheSlotToNomor = rows.mapNotNull { r ->
                    val id = extractSlotId(r.lokasi) ?: "S${r.nomor}"
                    val nomor = r.nomor
                    if (nomor != null) id to nomor else null
                }.toMap()

            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal memuat data"
            } finally {
                _loading.value = false
            }
        }
    }

    fun applyBeaconDetection(rawLocation: String?, currentUserIdOverride: Int? = null) {
        val newLoc = rawLocation?.trim()

        if (newLoc == null) {
            println("ℹ️ BEACON → User menjauh dari slot, abaikan.")
            return
        }

        viewModelScope.launch {
            try {
                val sess = sessionDao.getSessionOnce()
                val userId = currentUserIdOverride ?: sess?.userId

                if (userId == null) {
                    _error.value = "User belum login"
                    return@launch
                }

                when {

                    // =======================
                    // 1️⃣ USER MASUK GATE_IN
                    // =======================
                    newLoc.equals("Gate_In", ignoreCase = true) -> {
                        println("🚪 User $userId masuk area (Gate_In)")
                        logUserActivity(userId, "Gate_In", "entered")
                        return@launch
                    }

                    // =======================
                    // 2️⃣ USER KELUAR (GATE_OUT)
                    // =======================
                    newLoc.equals("Gate_Out", ignoreCase = true) -> {
                        println("🚪 User $userId keluar area parkir (Gate_Out)")

                        // fallback: cari slot occupied jika lastOccupiedSlot hilang
//                        val slotToFree = lastOccupiedSlotId
//                            ?: _statusById.value.entries
//                                .firstOrNull { it.value == "occupied" }?.key
//
//                        if (slotToFree != null) {
//                            println("🔓 Membebaskan slot $slotToFree (Gate_Out)")
//                            updateSlotStatus(slotToFree, "available", 0) // Opsi A
//                            _statusById.value = _statusById.value
//                                .toMutableMap().apply {
//                                    this[slotToFree] = "available"
//                                }
//                        } else {
//                            println("⚠ Tidak ada slot yang bisa dibebaskan saat Gate_Out")
//                        }

                        logUserActivity(userId, "Gate_Out", "exited")

                        lastOccupiedSlotId = null
                        reload()
                        return@launch
                    }

                    // =======================
                    // 3️⃣ USER DI SLOT PARKIR (S1..S5)
                    // =======================
                    newLoc.startsWith("S", ignoreCase = true) -> {

                        val slotStatus = _statusById.value[newLoc]

                        // Jika slot memang sudah ditempati user → abaikan
                        if (slotStatus == "occupied" && lastOccupiedSlotId == newLoc) {
                            println("ℹ️ Slot $newLoc sudah occupied oleh user ini, abaikan.")
                            return@launch
                        }

                        println("🚗 User $userId parkir di $newLoc")

                        // Bebaskan slot lama jika beda
                        lastOccupiedSlotId?.let { oldSlot ->
                            if (oldSlot != newLoc) {
                                println("🔄 Bebaskan slot lama $oldSlot")
                                updateSlotStatus(oldSlot, "available", userId)
                            }
                        }

                        // Update slot baru
                        updateSlotStatus(newLoc, "occupied", userId)
                        logUserActivity(userId, newLoc, "parking")

                        _statusById.value = _statusById.value.toMutableMap()
                            .apply { this[newLoc] = "occupied" }

                        lastOccupiedSlotId = newLoc.uppercase()   // <-- PENTING

                        reload()
                        return@launch
                    }

                    else -> {
                        println("ℹ️ Deteksi area $newLoc diabaikan.")
                        return@launch
                    }
                }

            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal update parkir"
            }
        }
    }

    private suspend fun logUserActivity(
        userId: Int,
        area: String,
        status: String,
        slotId: String? = null
    ) {
        try {
            if (cacheSlotToNomor.isEmpty()) reload()

            val resp = logRepo.sendLog(
                userid = userId,
                area = area,
                status = status
            )

            if (!resp.isSuccessful) {
                println("⚠️ Gagal mencatat aktivitas: ${resp.code()} ${resp.message()}")
            } else {
                println("✅ Log aktivitas terkirim: $status di $area")
            }
        } catch (e: Exception) {
            println("⚠️ Error kirim log aktivitas: ${e.localizedMessage}")
        }
    }

    private suspend fun updateSlotStatus(
        slotId: String,
        status: String,
        userId: Int
    ) {
        var nomor: Int? = cacheSlotToNomor[slotId]

        if (nomor == null) {
            nomor = findNomorByLokasi(slotId)
            if (nomor != null) {
                cacheSlotToNomor = cacheSlotToNomor.toMutableMap().apply {
                    put(slotId, nomor!!)
                }
            }
        }

        requireNotNull(nomor) { "Slot $slotId tidak ditemukan di database" }

        val resp = repo.updateParking(
            nomor = nomor!!,
            status = status,
            userID = userId   // <-- Opsi A: tidak null
        )

        if (!resp.isSuccessful) {
            throw Exception("Update slot $slotId gagal: ${resp.code()} ${resp.message()}")
        }
    }

    private suspend fun findNomorByLokasi(slotId: String): Int? {
        val resp = repo.getParkings()
        if (!resp.isSuccessful) return null

        val rows = resp.body().orEmpty()
        rows.firstOrNull { it.lokasi.equals(slotId, ignoreCase = true) }
            ?.nomor?.let { return it }

        for (r in rows) {
            val token = extractSlotId(r.lokasi)
            if (token != null && token.equals(slotId, ignoreCase = true))
                return r.nomor
        }

        return null
    }

    private fun extractSlotId(lokasi: String?): String? {
        if (lokasi.isNullOrBlank()) return null
        val rx = Regex("""\b([SD]\d+|S\d+)\b""", RegexOption.IGNORE_CASE)
        return rx.find(lokasi)?.value?.uppercase()
    }

    fun forceOccupySlotForDebug(slotId: String, userIdOverride: Int? = null) {
        applyBeaconDetection(slotId, currentUserIdOverride = userIdOverride)
    }
}

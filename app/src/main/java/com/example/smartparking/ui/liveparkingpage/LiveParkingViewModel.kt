package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.Parking
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.LogActivityRepository
import com.example.smartparking.data.repository.ParkingRepository
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

    private var releaseJob: kotlinx.coroutines.Job? = null
    private val releaseDelayMs = 15_000L

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

                // Build map status
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

    fun applyBeaconDetection(slotId: String?, currentUserIdOverride: Int? = null) {
        val id = slotId?.trim()?.takeIf {
            it.isNotEmpty() && !it.equals("Belum terdeteksi", true)
        } ?: return

        // Hindari spam deteksi slot yang sama
        if (id == lastOccupiedSlotId) return

        viewModelScope.launch {
            try {
                val sess = sessionDao.getSessionOnce()
                val userId = currentUserIdOverride ?: sess?.userId
                val role = sess?.role

                if (userId == null) {
                    _error.value = "User belum login"
                    return@launch
                }

                when {
                    // 🟢 GATE IN — hanya catat log masuk area
                    id.equals("Gate_In", ignoreCase = true) -> {
                        logUserActivity(userId, "gate_in", "entered")
                        println("🚪 User $userId masuk area parkir (Gate_In)")
                    }

                    // 🔴 GATE OUT — tandai keluar dan bebaskan slot sebelumnya
                    id.equals("Gate_Out", ignoreCase = true) -> {
                        // jika sebelumnya ada slot ditempati, bebaskan
                        lastOccupiedSlotId?.let { oldSlotId ->
                            updateSlotStatus(
                                slotId = oldSlotId,
                                status = "available",
                                userId = null
                            )
                            _statusById.value = _statusById.value.toMutableMap().apply {
                                this[oldSlotId] = "available"
                            }
                        }

                        logUserActivity(userId, "gate_out", "exited")
                        println("🚪 User $userId keluar area parkir (Gate_Out)")

                        lastOccupiedSlotId = null
                        reload()
                    }

                    // 🚗 SLOT PARKIR (S1–S5)
                    id.startsWith("S", ignoreCase = true) -> {
                        // Pastikan user tidak sudah parkir di slot lain

                        val activeSlot = _statusById.value.entries.find { (_, status) ->
                            status == "occupied"
                        }

                        if (activeSlot != null && activeSlot.key != id) {
                            println("⚠️ User $userId sudah parkir di ${activeSlot.key}, bebaskan dulu.")
                            updateSlotStatus(
                                slotId = activeSlot.key,
                                status = "available",
                                userId = userId
                            )
                        }

                        // Update slot baru jadi occupied
                        updateSlotStatus(
                            slotId = id,
                            status = "occupied",
                            userId = userId
                        )

                        logUserActivity(userId, id, "parking")

                        _statusById.value = _statusById.value.toMutableMap().apply {
                            this[id] = "occupied"
                        }

                        lastOccupiedSlotId = id
                        println("✅ User $userId parkir di $id")
                        reload()
                    }

                    // 🔹 Selain itu abaikan
                    else -> {
                        println("ℹ️ Deteksi area $id diabaikan (bukan gate atau slot).")
                    }
                }
            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal memperbarui status parkir"
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

            val slotKey = slotId ?: area
            val nomorFromCache = cacheSlotToNomor[slotKey]
            val nomor = nomorFromCache ?: findNomorByLokasi(slotKey)

            val resp = logRepo.sendLog(
                userid = userId,
                area = area,
                status = status
            )

            if (!resp.isSuccessful) {
                println("⚠️ Gagal mencatat aktivitas: ${resp.code()} ${resp.message()}")
            } else {
                println("✅ Log aktivitas terkirim: $status di $area (nomor=$nomor)")
            }
        } catch (e: Exception) {
            println("⚠️ Error kirim log aktivitas: ${e.localizedMessage}")
        }
    }

    private suspend fun updateSlotStatus(
        slotId: String,
        status: String,
        userId: Int?
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
            userID = userId
        )

        if (!resp.isSuccessful) {
            throw Exception("Update slot $slotId gagal: ${resp.code()} ${resp.message()}")
        }
    }

    private suspend fun findNomorByLokasi(slotId: String): Int? {
        val resp = repo.getParkings()
        if (!resp.isSuccessful) return null
        val rows = resp.body().orEmpty()

        rows.firstOrNull { it.lokasi.equals(slotId, ignoreCase = true) }?.nomor?.let { return it }
        for (r in rows) {
            val token = extractSlotId(r.lokasi)
            if (token != null && token.equals(slotId, ignoreCase = true)) return r.nomor
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

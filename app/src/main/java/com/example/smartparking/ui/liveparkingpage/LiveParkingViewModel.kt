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

    /** Map slotId ("S1", "S2") -> status ("occupied"/"available") */
    private val _statusById = MutableStateFlow<Map<String, String>>(emptyMap())
    val statusById: StateFlow<Map<String, String>> = _statusById

    /** cache slotId -> nomor */
    private var cacheSlotToNomor: Map<String, Int> = emptyMap()

    private var releaseJob: kotlinx.coroutines.Job? = null
    private val releaseDelayMs = 15_000L

    private var lastOccupiedSlotId: String? = null

    init {
        reload()
    }

    /** 🔹 Ambil semua data & bangun peta status + cache slot->nomor */
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

                // Build cache slotId->nomor
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

    /**
     * 🔹 Dipanggil saat Beacon mendeteksi slot baru
     *  - Update slot lama jadi available
     *  - Update slot baru jadi occupied
     *  - Catat log aktivitas user
     */
    fun applyBeaconDetection(slotId: String?, currentUserIdOverride: Int? = null) {
        val id = slotId?.trim()?.takeIf {
            it.isNotEmpty() && !it.equals("Belum terdeteksi", true)
        } ?: return

        if (id == lastOccupiedSlotId) return // Hindari spam deteksi slot sama

        println("Detect slot $id dengan tipe data ${id::class.simpleName}")

        viewModelScope.launch {
            try {
                val sess = sessionDao.getSessionOnce()
                val userId = currentUserIdOverride ?: sess?.userId

                if (userId == null) {
                    _error.value = "User belum login"
                    return@launch
                }

                // 🔸 Jika sebelumnya ada slot yang ditempati
                lastOccupiedSlotId?.let { oldSlotId ->
                    updateSlotStatus(
                        slotId = oldSlotId,
                        status = "available",
                        userId = null
                    )

                    // Log bahwa user keluar dari slot (moving)
                    logUserActivity(
                        userId = userId,
                        area = "in_area",
                        status = "moving",
                        slotId = oldSlotId
                    )

                    _statusById.value = _statusById.value.toMutableMap().apply {
                        this[oldSlotId] = "available"
                    }
                }

                // 🔸 Update slot baru jadi occupied
                updateSlotStatus(
                    slotId = id,
                    status = "occupied",
                    userId = userId
                )

                // Log bahwa user sedang parkir
                logUserActivity(
                    userId = userId,
                    area = id,
                    status = "parking",
                    slotId = id
                )

                _statusById.value = _statusById.value.toMutableMap().apply {
                    this[id] = "occupied"
                }

                lastOccupiedSlotId = id
                reload()

            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal memperbarui status parkir"
            }
        }
    }

    /** 🔹 Catat aktivitas user ke backend */
    private suspend fun logUserActivity(
        userId: Int,
        area: String,
        status: String,
        slotId: String? = null
    ) {
        try {
            // Reload cache jika kosong (antisipasi race condition)
            if (cacheSlotToNomor.isEmpty()) reload()

            // Gunakan slotId sebagai referensi pencarian nomor
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

    /** 🔹 Update status slot di database */
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

    /** 🔹 Cari nomor slot dari lokasi di DB */
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

    /** 🔹 Ekstrak SlotId dari lokasi (misal: "Slot S1" → "S1") */
    private fun extractSlotId(lokasi: String?): String? {
        if (lokasi.isNullOrBlank()) return null
        val rx = Regex("""\b([SD]\d+|S\d+)\b""", RegexOption.IGNORE_CASE)
        return rx.find(lokasi)?.value?.uppercase()
    }

    /** 🔹 Untuk debug manual (developer button) */
    fun forceOccupySlotForDebug(slotId: String, userIdOverride: Int? = null) {
        applyBeaconDetection(slotId, currentUserIdOverride = userIdOverride)
    }
}

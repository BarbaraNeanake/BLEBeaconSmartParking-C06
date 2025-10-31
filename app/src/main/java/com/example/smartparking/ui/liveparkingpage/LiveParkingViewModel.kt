package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartparking.data.model.Parking
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.ParkingRepository
import com.example.smartparking.data.repository.dao.SessionDao
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class LiveParkingViewModel(
    private val repo: ParkingRepository = ParkingRepository(RetrofitProvider.parkingApi),
    private val sessionDao : SessionDao
) : ViewModel() {

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    /** peta: slotId ("S1", "S2") -> "occupied"/"available"/"disabled_slot" */
    private val _statusById = MutableStateFlow<Map<String, String>>(emptyMap())
    val statusById: StateFlow<Map<String, String>> = _statusById

    /** cache mapping untuk mempercepat lookup */
    private var cacheSlotToNomor: Map<String, Int> = emptyMap()


    private var releaseJob: kotlinx.coroutines.Job? = null
    private val releaseDelayMs = 15_000L // 15 detik grace period

    private var lastOccupiedSlotId: String? = null

    init { reload() }

    /** Ambil semua data & bangun peta status + cache slot->nomor */
    fun reload() {
        if (_loading.value) return
        viewModelScope.launch {
            _loading.value = true
            _error.value = null
            try {
                val resp = repo.getParkings()
                if (!resp.isSuccessful) throw Exception("HTTP ${resp.code()} ${resp.message()}")

                val rows: List<Parking> = resp.body().orEmpty()

                // Bangun statusById (slotId akan diambil dari row.lokasi atau fallback S{nomor})
                val statusMap: Map<String, String> = rows.associate { row: Parking ->
                    val id = extractSlotId(row.lokasi) ?: "S${row.nomor}"
                    id to (row.status?.lowercase() ?: "available")
                }
                _statusById.value = statusMap

                // Cache slotId -> nomor untuk update cepat
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
     * Dipanggil saat Beacon mendeteksi slot: "S3", "S1", dst.
     * - Free slot lama (jika ada)
     * - Occupy slot baru
     * - Refresh status
     */
    fun applyBeaconDetection(slotId: String?, currentUserIdOverride: Int? = null) {
        val id = slotId?.trim()?.takeIf {
            it.isNotEmpty() && !it.equals("Belum terdeteksi", true)
        } ?: return

        // Hindari spam (tidak usah update kalau slot sama)
        if (id == lastOccupiedSlotId) return

        viewModelScope.launch {
            try {
                // 1️⃣ Ambil data user aktif dari session
                val sess = sessionDao.getSessionOnce()
                val userId = currentUserIdOverride ?: when (val v = sess?.userId) {
                    is Int -> v
                    is Long -> v.toInt()
                    else -> null
                }
                val role = sess?.role

                // 2️⃣ Jika sebelumnya ada slot yang ditempati
                lastOccupiedSlotId?.let { oldSlotId ->
                    // Update slot lama jadi available
                    updateSlotStatus(
                        slotId = oldSlotId,
                        status = "available",
                        userId = null,
                        rolesUser = null
                    )

                    // Update UI langsung (optimistik)
                    _statusById.value = _statusById.value.toMutableMap().apply {
                        this[oldSlotId] = "available"
                    }
                }

                // 3️⃣ Update slot baru jadi occupied
                updateSlotStatus(
                    slotId = id,
                    status = "occupied",
                    userId = userId,
                    rolesUser = role
                )

                // Update UI langsung (optimistik)
                _statusById.value = _statusById.value.toMutableMap().apply {
                    this[id] = "occupied"
                }

                // 4️⃣ Simpan slot terakhir
                lastOccupiedSlotId = id

                // 5️⃣ Refresh data dari server untuk sinkronisasi
                reload()

            } catch (e: Exception) {
                _error.value = e.message ?: "Gagal memperbarui status parkir"
            }
        }
    }


    private fun optimisticSwitch(oldId: String?, newId: String) {
        val m = _statusById.value.toMutableMap()
        oldId?.let { if (m.containsKey(it)) m[it] = "available" }
        m[newId] = "occupied"
        _statusById.value = m
    }

    private fun scheduleReleaseIfIdle() {
        if (releaseJob?.isActive == true) return
        val oldId = lastOccupiedSlotId ?: return
        releaseJob = viewModelScope.launch {
            kotlinx.coroutines.delay(releaseDelayMs)
            // Setelah 15 dtk tetap tidak ada deteksi, rilis slot lama
            try {
                updateSlotStatus(oldId, status = "available", userId = null, rolesUser = null)
            } catch (_: Exception) { /* log kalau perlu */ }
            lastOccupiedSlotId = null
            // Optimistik: set hijau di UI
            val m = _statusById.value.toMutableMap()
            m[oldId] = "available"
            _statusById.value = m
        }
    }

    private fun cancelRelease() {
        releaseJob?.cancel()
        releaseJob = null
    }

    // ====== Sudah kamu punya, tambahkan rolesUser ======
    private suspend fun updateSlotStatus(slotId: String, status: String, userId: Int?, rolesUser: String?) {
        var nomor: Int? = cacheSlotToNomor[slotId]
        if (nomor == null) {
            nomor = findNomorByLokasi(slotId)
            if (nomor != null) {
                cacheSlotToNomor = cacheSlotToNomor.toMutableMap().apply { put(slotId, nomor!!) }
            }
        }
        requireNotNull(nomor) { "Slot $slotId tidak ditemukan di database" }

        val resp = repo.updateParking(
            nomor = nomor!!,
            status = status,
            userID = userId,
            rolesUser = rolesUser
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

    // LiveParkingViewModel.kt
    fun forceOccupySlotForDebug(slotId: String, userIdOverride: Int? = null) {
        applyBeaconDetection(slotId, currentUserIdOverride = userIdOverride)
    }

}

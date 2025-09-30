package com.example.smartparking.ui.liveparkingpage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class Slot(
    val id: String,
    val xPct: Float,   // posisi relatif 0..1
    val yPct: Float,
    val wPct: Float,
    val hPct: Float,
    val occupied: Boolean // true = merah (terisi), false = hijau (kosong)
)

data class Lot(
    val name: String,
    val imageRes: Int,
    val slots: List<Slot>
) {
    val used: Int get() = slots.count { it.occupied }
    val free: Int get() = slots.size - used
}

data class LiveParkingUiState(
    val loading: Boolean = true,
    val lots: List<Lot> = emptyList(),
    val error: String? = null
)

class LiveParkingViewModel : ViewModel() {

    private val _ui = MutableStateFlow(LiveParkingUiState())
    val ui: StateFlow<LiveParkingUiState> = _ui

    init {
        // Dummy loader (simulasi fetch API)
        viewModelScope.launch {
            delay(300)
            _ui.value = LiveParkingUiState(
                loading = false,
                lots = dummyLots()
            )
        }
    }

    /** Nanti ganti ambil dari Repository (Retrofit/Ktor) */
    private fun dummyLots(): List<Lot> = listOf(
        Lot(
            name = "Lapangan Satu Bumi",
            imageRes = com.example.smartparking.R.drawable.homepage_map,
            slots = listOf(
                Slot("A1", 0.78f, 0.08f, 0.14f, 0.07f, occupied = false),
                Slot("A2", 0.78f, 0.18f, 0.14f, 0.07f, occupied = true),
                Slot("A3", 0.78f, 0.28f, 0.14f, 0.07f, occupied = false),
                Slot("A4", 0.78f, 0.38f, 0.14f, 0.07f, occupied = false),
                Slot("A5", 0.78f, 0.48f, 0.14f, 0.07f, occupied = true),
            )
        ),
        Lot(
            name = "Depan DTETI",
            imageRes = com.example.smartparking.R.drawable.homepage_map,
            slots = listOf(
                Slot("B1", 0.07f, 0.10f, 0.14f, 0.07f, occupied = false),
                Slot("B2", 0.07f, 0.20f, 0.14f, 0.07f, occupied = false),
                Slot("B3", 0.07f, 0.30f, 0.14f, 0.07f, occupied = true),
                Slot("B4", 0.07f, 0.40f, 0.14f, 0.07f, occupied = true),
                Slot("B5", 0.45f, 0.70f, 0.12f, 0.07f, occupied = true),
                Slot("B6", 0.58f, 0.70f, 0.12f, 0.07f, occupied = true),
                Slot("B7", 0.71f, 0.70f, 0.12f, 0.07f, occupied = true),
            )
        )
    )
}

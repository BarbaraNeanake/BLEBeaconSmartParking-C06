package com.example.smartparking.ui.historypage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class HistoryItem(
    val date: String,    // contoh: "13-09-2025"
    val time: String,    // contoh: "09.00"
    val location: String // contoh: "B3 DTETI FT UGM"
)

data class HistoryUiState(
    val loading: Boolean = true,
    val name: String = "Barbara Neanake",
    val items: List<HistoryItem> = emptyList(),
    val error: String? = null
)

class HistoryViewModel : ViewModel() {

    private val _ui = MutableStateFlow(HistoryUiState())
    val ui: StateFlow<HistoryUiState> = _ui

    init {
        // Dummy fetch (simulasi BE)
        viewModelScope.launch {
            delay(300)
            _ui.value = HistoryUiState(
                loading = false,
                name = "Barbara Neanake",
                items = listOf(
                    HistoryItem("13-09-2025", "09.00", "B3 DTETI FT UGM"),
                    HistoryItem("13-09-2025", "10.30", "Lapangan Satu Bumi"),
                    HistoryItem("12-09-2025", "13.15", "DTGL Lingkar Teknik"),
                    HistoryItem("11-09-2025", "08.45", "DTMI Bawah"),
                )
            )
        }
    }
}
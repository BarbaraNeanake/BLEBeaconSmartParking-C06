package com.example.smartparking.ui.historypage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class HistoryItem(
    val date: String,    // "13-09-2025"
    val time: String,    // "09.00 AM"
    val location: String // "B3 DTETI FT UGM"
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
        // Dummy loader
        viewModelScope.launch {
            delay(300)
            _ui.value = HistoryUiState(
                loading = false,
                name = "Barbara Neanake",
                items = listOf(
                    HistoryItem("13-09-2025", "09.00 AM", "B3 DTETI FT UGM"),
                    HistoryItem("13-09-2025", "10.30 AM", "Lapangan Satu Bumi"),
                    HistoryItem("12-09-2025", "01.15 PM", "DTGL Lingkar Teknik"),
                    HistoryItem("11-09-2025", "08.45 AM", "DTMI Bawah"),
                )
            )
        }
    }
}

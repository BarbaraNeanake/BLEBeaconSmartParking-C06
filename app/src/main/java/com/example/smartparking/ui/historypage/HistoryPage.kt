package com.example.smartparking.ui.historypage

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import com.example.smartparking.data.repository.LogActivityRepository
import com.example.smartparking.data.repository.dao.SessionDao
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

@Composable
fun HistoryPage(
    logRepository: LogActivityRepository,
    sessionDao: SessionDao
) {
    val vm: HistoryViewModel = viewModel(
        factory = HistoryViewModelFactory(logRepository, sessionDao)
    )
    val ui by vm.ui.collectAsState()
    HistoryContent(ui, onRetry = { vm.retry() })
}

@Composable
private fun HistoryContent(
    ui: HistoryUiState,
    onRetry: () -> Unit
) {
    val bg = Brush.verticalGradient(
        listOf(GradientTop.copy(alpha = 0.92f), Color.White, GradientBottom.copy(alpha = 0.92f))
    )

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .systemBarsPadding()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(bottom = 24.dp)
    ) {
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Image(
                    painter = painterResource(R.drawable.ugm_logo),
                    contentDescription = "UGM Logo",
                    modifier = Modifier.size(80.dp)
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Riwayat Aktivitas Parkir\nFakultas Teknik UGM",
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.titleLarge.copy(
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 18.sp,
                        lineHeight = 22.sp
                    )
                )
            }
        }
        when {
            ui.loading -> {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 80.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                }
            }
            ui.error != null -> {
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 50.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Error: ${ui.error}",
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center
                        )
                        Spacer(Modifier.height(10.dp))
                        Button(onClick = onRetry) {
                            Text("Coba Lagi")
                        }
                    }
                }
            }
            else -> {
                item {
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        elevation = CardDefaults.cardElevation(6.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 14.dp, horizontal = 16.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Histori Parkir • ${ui.name}",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.titleMedium.copy(
                                    fontWeight = FontWeight.SemiBold
                                )
                            )
                        }
                    }
                }
                item {
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        elevation = CardDefaults.cardElevation(6.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .padding(14.dp)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp, horizontal = 6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Tanggal", fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center, modifier = Modifier.weight(1.2f))
                                Text("Lokasi", fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center, modifier = Modifier.weight(0.8f))
                                Text("Status", fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center, modifier = Modifier.weight(1.0f))
                            }
                            Divider()
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(min = 120.dp, max = 340.dp)
                                    .padding(horizontal = 6.dp)
                            ) {
                                ui.items.forEach { log ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 10.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        val (date, timePart) = extractDateTimeParts(log.time)
                                        Column(
                                            modifier = Modifier.weight(1.2f),
                                            horizontalAlignment = Alignment.CenterHorizontally
                                        ) {
                                            Text(date, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                                            Text(timePart, style = MaterialTheme.typography.bodySmall.copy(color = Color.Gray))
                                        }
                                        Text(log.area?: "-", textAlign = TextAlign.Center, modifier = Modifier.weight(0.8f))
                                        val color = when (log.status?.lowercase()) {
                                            "masuk" -> Color(0xFF4CAF50)
                                            "keluar" -> Color(0xFFF44336)
                                            else -> Color.Gray
                                        }
                                        Text(
                                            log.status?.uppercase() ?: "-",
                                            textAlign = TextAlign.Center,
                                            color = color,
                                            modifier = Modifier.weight(1.0f)
                                        )
                                    }
                                    Divider()
                                }
                            }
                        }
                    }
                }
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 12.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Image(
                            painter = painterResource(R.drawable.icon_historypage),
                            contentDescription = null,
                            modifier = Modifier.size(200.dp)
                        )
                    }
                }
            }
        }
    }
}


private fun formatTime(time: Any?): String {
    if (time == null) return "-"
    return try {
        val raw = time.toString().replace("T", " ")
        val cleaned = raw.split(".").firstOrNull() ?: raw

        val parsed = LocalDateTime.parse(cleaned, DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
        val wibTime = parsed.atZone(ZoneId.of("UTC")).withZoneSameInstant(ZoneId.of("Asia/Jakarta"))
        wibTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
    } catch (e: Exception) {
        time.toString().split(".").firstOrNull() ?: time.toString()
    }
}

private fun extractDateTimeParts(time: Any?): Pair<String, String> {
    if (time == null) return "-" to "-"
    val raw = when (time) {
        is String -> time
        else -> time.toString()
    }
    val clean = raw.split(".").firstOrNull() ?: raw
    return if (clean.contains("T")) {
        val (date, timePart) = clean.split("T")
        date to timePart
    } else if (clean.contains(" ")) {
        val (date, timePart) = clean.split(" ")
        date to timePart
    } else clean to "-"
}

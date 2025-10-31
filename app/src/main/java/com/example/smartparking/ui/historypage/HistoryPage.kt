package com.example.smartparking.ui.historypage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun HistoryPage(vm: HistoryViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    HistoryContent(ui)
}

/** UI murni yang bisa dipreview/dipanggil runtime */
@Composable
private fun HistoryContent(ui: HistoryUiState) {
    val bg = Brush.verticalGradient(
        listOf(GradientTop.copy(alpha = 0.92f), Color.White, GradientBottom.copy(alpha = 0.92f))
    )

    // Page scroll keseluruhan
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(bottom = 24.dp)
    ) {
        // Header: logo + title
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Image(
                    painter = painterResource(R.drawable.ugm_logo),
                    contentDescription = "UGM Logo",
                    modifier = Modifier.size(60.dp)
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Parking History\n di Fakultas Teknik UGM",
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.titleLarge.copy(
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 18.sp,
                        lineHeight = 22.sp
                    )
                )
            }
        }

        // Loading / error
        when {
            ui.loading -> {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 80.dp),
                        contentAlignment = Alignment.Center
                    ) { CircularProgressIndicator() }
                }
            }
            ui.error != null -> {
                item {
                    Text(
                        text = "Error: ${ui.error}",
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center
                    )
                }
            }
            else -> {
                // Card tabel
                item {
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        elevation = CardDefaults.cardElevation(6.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(Modifier.fillMaxWidth().padding(14.dp)) {
                            Text(
                                text = "Histori Parkir • ${ui.name}",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.titleMedium.copy(
                                    fontWeight = FontWeight.SemiBold
                                )
                            )
                            Spacer(Modifier.height(8.dp))

                            // Header row
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp, horizontal = 6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Tanggal", fontWeight = FontWeight.SemiBold,
                                    modifier = Modifier.weight(1.1f))
                                Text("Waktu", fontWeight = FontWeight.SemiBold,
                                    modifier = Modifier.weight(0.9f))
                                Text("Lokasi Parkir", fontWeight = FontWeight.SemiBold,
                                    modifier = Modifier.weight(1.4f))
                            }
                            Divider()

                            // List data (scroll pada page — tidak nested scroll)
                            LazyColumn(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(min = 120.dp, max = 340.dp),
                            ) {
                                items(ui.items) { item ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 10.dp, horizontal = 6.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(item.date, modifier = Modifier.weight(1.1f))
                                        Text(item.time, modifier = Modifier.weight(0.9f))
                                        Text(item.location, modifier = Modifier.weight(1.4f))
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

/* ================= PREVIEW ================= */
@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "History – Light")
@Composable private fun PreviewHistoryLight() {
    SmartParkingTheme {
        HistoryContent(
            HistoryUiState(
                loading = false,
                name = "Barbara Neanake",
                items = listOf(
                    HistoryItem("13-09-2025", "09.00", "B3 DTETI FT UGM"),
                    HistoryItem("13-09-2025", "10.30", "Lapangan Satu Bumi"),
                    HistoryItem("12-09-2025", "13.15", "DTGL Lingkar Teknik"),
                    HistoryItem("11-09-2025", "08.45", "DTMI Bawah"),
                )
            )
        )
    }
}
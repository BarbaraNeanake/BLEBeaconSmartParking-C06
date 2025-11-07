package com.example.smartparking.ui.pelanggaranpage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
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
fun PelanggaranPage(
    vm: PelanggaranViewModel = viewModel()
) {
    val ui by vm.uiState.collectAsState()

    LaunchedEffect(Unit) {
        vm.refresh()
    }

    PelanggaranContent(ui = ui)
}

@Composable
private fun PelanggaranContent(ui: PelanggaranUiState) {
    val bg = Brush.verticalGradient(
        listOf(
            GradientTop.copy(alpha = 0.92f),
            Color.White,
            GradientBottom.copy(alpha = 0.92f)
        )
    )

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(bottom = 24.dp)
    ) {
        // HEADER
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Image(
                    painter = painterResource(R.drawable.ugm_logo),
                    contentDescription = "UGM Logo",
                    modifier = Modifier.size(60.dp),
                    contentScale = ContentScale.Fit
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Pelanggaran Kendaraan Roda 4\nFakultas Teknik UGM",
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
                    Text(
                        text = "Error: ${ui.error}",
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center
                    )
                }
            }

            else -> {
                // CARD JUDUL
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
                                text = "Pelanggaran • ${ui.ownerName}",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.titleMedium.copy(
                                    fontWeight = FontWeight.SemiBold
                                )
                            )
                        }
                    }
                }

                // CARD TABEL (NAVY)
                item {
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        elevation = CardDefaults.cardElevation(6.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFF0A2342)   // navy
                        ),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .padding(14.dp)
                        ) {
                            // header
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp, horizontal = 6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "No",
                                    fontWeight = FontWeight.SemiBold,
                                    textAlign = TextAlign.Center,
                                    color = Color.White,
                                    modifier = Modifier.weight(0.35f)
                                )
                                Text(
                                    "Tanggal",
                                    fontWeight = FontWeight.SemiBold,
                                    textAlign = TextAlign.Center,
                                    color = Color.White,
                                    modifier = Modifier.weight(0.9f)
                                )
                                Text(
                                    "Slot",
                                    fontWeight = FontWeight.SemiBold,
                                    textAlign = TextAlign.Center,
                                    color = Color.White,
                                    modifier = Modifier.weight(0.7f)
                                )
                                Text(
                                    "Jenis Pelanggaran",
                                    fontWeight = FontWeight.SemiBold,
                                    textAlign = TextAlign.Center,
                                    color = Color.White,
                                    modifier = Modifier.weight(1.4f)
                                )
                            }
                            Divider(color = Color(0x33FFFFFF))

                            // LIST DATA — mengikuti jumlah data dari DB
                            LazyColumn(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(min = 120.dp, max = 340.dp),
                            ) {
                                itemsIndexed(ui.records) { idx, item ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 10.dp, horizontal = 6.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            (idx + 1).toString(),
                                            textAlign = TextAlign.Center,
                                            color = Color.White,
                                            modifier = Modifier.weight(0.35f)
                                        )
                                        Text(
                                            item.date,
                                            textAlign = TextAlign.Center,
                                            color = Color.White,
                                            modifier = Modifier.weight(0.9f)
                                        )
                                        Text(
                                            item.slotCode,
                                            textAlign = TextAlign.Center,
                                            color = Color.White,
                                            modifier = Modifier.weight(0.7f)
                                        )
                                        Text(
                                            item.violationType,
                                            textAlign = TextAlign.Start,
                                            color = Color.White,
                                            modifier = Modifier.weight(1.4f)
                                        )
                                    }
                                    Divider(color = Color(0x22FFFFFF))
                                }
                            }
                        }
                    }
                }

                // GAMBAR DI BAWAH TABEL
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 12.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Image(
                            painter = painterResource(R.drawable.icon_pelanggaranpage),
                            contentDescription = null,
                            modifier = Modifier.size(200.dp),
                            contentScale = ContentScale.Fit
                        )
                    }
                }
            }
        }
    }
}

/* ===== PREVIEW ===== */
@Preview(
    showBackground = true,
    uiMode = Configuration.UI_MODE_NIGHT_NO,
    name = "Pelanggaran – Light"
)
@Composable
private fun PreviewPelanggaranLight() {
    SmartParkingTheme(darkTheme = false, dynamicColor = false) {
        PelanggaranContent(
            PelanggaranUiState(
                loading = false,
                ownerName = "Barbara Neanake",
                records = listOf(
                    PelanggaranRecord("07-11-2025", "P3", "Parkir di luar garis"),
                    PelanggaranRecord("05-11-2025", "P12", "Menggunakan slot disabilitas"),
                    PelanggaranRecord("01-11-2025", "P7", "Parkir melebihi durasi"),
                    PelanggaranRecord("30-10-2025", "P1", "Parkir di area terlarang"),
                )
            )
        )
    }
}

package com.example.smartparking.ui.homepage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

// ---------- Data tabel ----------
data class ParkingLocation(
    val id: String,
    val location: String,
    val estimatedCapacity: Int
)

private val parkingLocations: List<ParkingLocation> = listOf(
    ParkingLocation("P1", "ERIC", 0),
    ParkingLocation("P2", "DTSL", 15),
    ParkingLocation("P3", "DTGD", 4),
    ParkingLocation("P4", "DTAP Arsitektur", 20),
    ParkingLocation("P5", "DTAP Perencanaan", 20),
    ParkingLocation("P6", "Lapangan Basket", 7),
    ParkingLocation("P7", "DTK Bawah", 5),
    ParkingLocation("P8", "DTK Taman Manufaktur", 7),
    ParkingLocation("P9", "DTETI Taman Manufaktur", 0),
    ParkingLocation("P10", "DTMI Taman Manufaktur", 0),
    ParkingLocation("P11", "DTMI Bawah", 50),
    ParkingLocation("P12", "DTETI Lingkar Teknik", 14),
    ParkingLocation("P13", "DTNTF", 8),
    ParkingLocation("P14", "DTGL Lingkar Teknik", 4),
    ParkingLocation("P15", "DTGL Dalam", 0),
    ParkingLocation("P16", "Satu Bumi", 0)
)

// ---------- UI ----------
@Composable
fun HomePage(
    vm: HomePageViewModel = viewModel()
) {
    val parkingStatus by vm.parkingStatus
    LaunchedEffect(Unit) { vm.fetchParkingStatus() }

    // background gradasi
    val bg = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.9f),
                Color.White,
                GradientBottom.copy(alpha = 0.9f)
            )
        )
    }

    val listState = rememberLazyListState()

    LazyColumn(
        state = listState,
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
        contentPadding = PaddingValues(bottom = 16.dp)
    ) {
        // Header
        item {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth()
            ) {
                Image(
                    painter = painterResource(id = R.drawable.ugm_logo),
                    contentDescription = "UGM Logo",
                    modifier = Modifier.size(80.dp),
                    contentScale = ContentScale.Fit
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = buildAnnotatedString {
                        append("Selamat Datang Dalam\n")
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                            append("Sistem Monitoring Kantong Parkir Roda 4\n")
                        }
                        append("Fakultas Teknik UGM")
                    },
                    style = MaterialTheme.typography.bodyLarge.copy(
                        fontSize = 16.sp,
                        textAlign = TextAlign.Center
                    ),
                    textAlign = TextAlign.Center
                )
            }
        }

        // Info slot
        item {
            Box(
                modifier = Modifier
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                InfoCard(
                    title = "Slot Parkir Mobil di FT Saat Ini",
                    textAlign = TextAlign.Center,
                    totalSlots = parkingStatus.totalSlots,
                    usedSlots = parkingStatus.usedSlots
                )
            }
        }

        // Peta
        item {
            Card(
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(6.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Image(
                    painter = painterResource(id = R.drawable.homepage_map),
                    contentDescription = "Peta FT",
                    contentScale = ContentScale.FillBounds,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(280.dp)
                )
            }
        }

        // Tabel kapasitas
        item {
            ParkingTable(parkingLocations)
        }
    }
}

@Composable
private fun InfoCard(title: String, totalSlots: Int, usedSlots: Int, textAlign: TextAlign) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(6.dp)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                title,
                style = MaterialTheme.typography.titleMedium.copy(
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 16.sp
                )
            )
            Spacer(Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatBox(label = "Tempat Tersedia", value = (totalSlots - usedSlots))
                VerticalDivider()
                StatBox(label = "Parkir Terpakai", value = usedSlots)
            }
        }
    }
}

@Composable
private fun StatBox(label: String, value: Int) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, fontSize = 14.sp)
        Text(
            value.toString(),
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun VerticalDivider() {
    Box(
        modifier = Modifier
            .height(40.dp)
            .width(1.dp)
            .background(Color(0x33000000))
    )
}

@Composable
private fun ParkingTable(rows: List<ParkingLocation>) {
    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(4.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.fillMaxWidth()) {
            // header
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFECECEC))
                    .padding(vertical = 10.dp, horizontal = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Kode", fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                Text("Lokasi/Departemen", fontWeight = FontWeight.Bold, modifier = Modifier.weight(2f))
                Text(
                    "Sisa Slot",
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.weight(1f)
                )
            }
            // rows
            rows.forEach { p: ParkingLocation ->
                Row(
                    Modifier
                        .fillMaxWidth()
                        .border(0.5.dp, Color.LightGray)
                        .padding(vertical = 10.dp, horizontal = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(p.id, modifier = Modifier.weight(1f))
                    Text(p.location, modifier = Modifier.weight(2f))
                    Text(
                        p.estimatedCapacity.toString(),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

/* ==================== Preview ==================== */
@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO)
@Composable
private fun PreviewHomePage() {
    SmartParkingTheme {
        HomePage()
    }
}

package com.example.smartparking.ui.liveparkingpage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun LiveParkingPage(vm: LiveParkingViewModel = viewModel()) {
    val state by vm.ui.collectAsStateWithLifecycle()

    val bg = remember {
        Brush.verticalGradient(listOf(GradientTop.copy(0.95f), Color.White, GradientBottom.copy(0.95f)))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            "Live Parking Map\nKantong Parkir Roda 4\nFakultas Teknik UGM",
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
        )

        if (state.loading) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
        } else if (state.error != null) {
            Text("Error: ${state.error}", color = MaterialTheme.colorScheme.error)
        } else {
            state.lots.forEachIndexed { index, lot ->
                LotCard(index + 1, lot)
            }
        }
    }
}

@Composable
private fun LotCard(index: Int, lot: Lot) {
    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(6.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("$index. ${lot.name}", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)

            // Gambar peta + overlay slot relatif
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
            ) {
                val boxW = maxWidth
                val boxH = maxHeight
                Image(
                    painter = painterResource(id = lot.imageRes),
                    contentDescription = lot.name,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )

                lot.slots.forEach { slot ->
                    val x = (boxW.value * slot.xPct).dp
                    val y = (boxH.value * slot.yPct).dp
                    val w = (boxW.value * slot.wPct).dp
                    val h = (boxH.value * slot.hPct).dp
                    val color = if (slot.occupied) Color(0xFFD93636) else Color(0xFF18B46E)

                    Box(
                        modifier = Modifier
                            .offset(x, y)
                            .size(w, h)
                            .background(color, RoundedCornerShape(5.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(slot.id, color = Color.White, style = MaterialTheme.typography.labelMedium)
                    }
                }
            }

            // Ringkasan tersedia/terpakai
            SummaryRow(free = lot.free, used = lot.used)
        }
    }
}

@Composable
private fun SummaryRow(free: Int, used: Int) {
    Card(shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth()) {
            Column(
                Modifier
                    .weight(1f)
                    .padding(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Tempat Tersedia", fontWeight = FontWeight.SemiBold)
                Text("$free", style = MaterialTheme.typography.titleMedium)
            }
            Divider(Modifier.width(1.dp).height(48.dp))
            Column(
                Modifier
                    .weight(1f)
                    .padding(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Parkir Terpakai", fontWeight = FontWeight.SemiBold)
                Text("$used", style = MaterialTheme.typography.titleMedium)
            }
        }
    }
}

@Preview(uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
private fun PreviewLiveParking() {
    SmartParkingTheme { LiveParkingPage() }
}

package com.example.smartparking.ui.liveparkingpage

import androidx.annotation.DrawableRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import com.example.smartparking.ui.beacontest.BeaconViewModel
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import kotlin.math.roundToInt

// ----------------------------
// Data model tetap (koordinat statis)
// ----------------------------
data class Slot(
    val id: String,
    val xPct: Float,
    val yPct: Float,
    val wPct: Float,
    val hPct: Float,
    val accessible: Boolean = false,
    val occupied: Boolean = false
)

data class Lot(
    val name: String,
    @DrawableRes val imageRes: Int,
    val free: Int,
    val used: Int,
    val slots: List<Slot>
)

// ----------------------------
// COMPOSABLE utama
// ----------------------------
@Composable
fun LiveParkingPage(
    vm: LiveParkingViewModel = viewModel(),
    beaconVM: BeaconViewModel = viewModel(),
    currentUserId: Int? = null
) {
    // State dari backend
    val loading by vm.loading.collectAsStateWithLifecycle()
    val error by vm.error.collectAsStateWithLifecycle()
    val statusById by vm.statusById.collectAsStateWithLifecycle()

    // State dari BLE
    val detectedSlot by beaconVM.detectedSlot.collectAsStateWithLifecycle()

    // Mulai BLE scanning otomatis
    DisposableEffect(Unit) {
        beaconVM.startScan()
        onDispose { beaconVM.stopScan() }
    }

    // Saat beacon mendeteksi slot baru → update backend
    LaunchedEffect(detectedSlot) {
        if (!detectedSlot.equals("Belum terdeteksi", ignoreCase = true)) {
            vm.applyBeaconDetection(detectedSlot, currentUserId)
        }
    }

    // Base lot dengan koordinat tetap
    val baseLot = remember { sampleLot(R.drawable.liveparkingmap) }

    // Update warna berdasarkan status backend
    val coloredLot = remember(baseLot, statusById) {
        val updatedSlots = baseLot.slots.map { s ->
            val status = statusById[s.id]?.lowercase()
            val isOcc = status == "occupied"
            val isAcc = status == "disabled_slot"
            s.copy(occupied = isOcc, accessible = isAcc)
        }
        val used = updatedSlots.count { it.occupied }
        val free = updatedSlots.size - used
        baseLot.copy(slots = updatedSlots, used = used, free = free)
    }

    // ----------------------------
    // UI: background gradasi + header (logo + judul center)
    // ----------------------------
    val bg = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.9f),
                Color.White,
                GradientBottom.copy(alpha = 0.9f)
            )
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Logo UGM (center)
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .size(84.dp)
                    .padding(top = 4.dp, bottom = 6.dp)
            )

            // Judul halaman (center)
            Text(
                text = "Live Parking",
                style = MaterialTheme.typography.titleLarge.copy(
                    fontWeight = FontWeight.Bold,
                    fontSize = 22.sp
                ),
                textAlign = TextAlign.Center
            )
            Text(
                text = "Kantong Parkir Fakultas Teknik UGM",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(14.dp))

            // Konten dinamis — TIDAK diubah
            when {
                loading -> CircularProgressIndicator()
                error != null -> {
                    Text(text = error ?: "-", color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = vm::reload, shape = RoundedCornerShape(12.dp)) { Text("Coba Lagi") }
                }
                else -> LotCard(lot = coloredLot, onRefresh = vm::reload)
            }
        }
    }
}

// ----------------------------
// CARD (TIDAK DIUBAH)
// ----------------------------
@Composable
private fun LotCard(lot: Lot, onRefresh: () -> Unit) {
    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(6.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.98f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(lot.name, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold))
                AssistChip(onClick = {}, label = { Text("${lot.free} kosong • ${lot.used} terpakai") })
            }

            Spacer(Modifier.height(10.dp))

            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
                    .clip(RoundedCornerShape(12.dp))
            ) {
                val density = LocalDensity.current
                val boxWidthPx = with(density) { maxWidth.toPx() }
                val boxHeightPx = with(density) { maxHeight.toPx() }

                Image(
                    painter = painterResource(lot.imageRes),
                    contentDescription = lot.name,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )

                lot.slots.forEach { s ->
                    val xPx = (boxWidthPx * s.xPct).coerceIn(0f, boxWidthPx - (boxWidthPx * s.wPct))
                    val yPx = (boxHeightPx * s.yPct).coerceIn(0f, boxHeightPx - (boxHeightPx * s.hPct))
                    val wDp = with(density) { (boxWidthPx * s.wPct).toDp() }
                    val hDp = with(density) { (boxHeightPx * s.hPct).toDp() }

                    val baseColor = when {
                        s.accessible -> Color(0xFF2F65F5)
                        s.occupied -> Color(0xFFD93636)
                        else -> Color(0xFF18B46E)
                    }

                    Box(
                        modifier = Modifier
                            .absoluteOffset { IntOffset(xPx.roundToInt(), yPx.roundToInt()) }
                            .size(width = wDp, height = hDp)
                            .background(baseColor.copy(alpha = 0.85f), RoundedCornerShape(6.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = s.id, style = MaterialTheme.typography.labelMedium, color = Color.White)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))
            Button(onClick = onRefresh, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                Text("Refresh")
            }
        }
    }
}

// ----------------------------
// KOORDINAT TETAP (TIDAK DIUBAH)
// ----------------------------
private fun sampleLot(@DrawableRes mapRes: Int): Lot {
    val slots = listOf(
        Slot("S1", 0.14f, 0.50f, 0.10f, 0.33f),
        Slot("S2", 0.30f, 0.50f, 0.10f, 0.33f),
        Slot("S3", 0.45f, 0.50f, 0.10f, 0.33f),
        Slot("S4", 0.61f, 0.50f, 0.10f, 0.33f),
        Slot("S5", 0.76f, 0.50f, 0.10f, 0.33f)
    )
    val used = slots.count { it.occupied }
    val free = slots.size - used
    return Lot(
        name = "Departemen",
        imageRes = mapRes,
        free = free,
        used = used,
        slots = slots
    )
}

@Composable
private fun DeveloperBar(
    onPick: (String) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        listOf("S1","S2","S3","S4","S5").forEach { id ->
            Button(onClick = { onPick(id) }) { Text(id) }
        }
    }
}

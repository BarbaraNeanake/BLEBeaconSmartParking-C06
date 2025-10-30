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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import kotlin.math.roundToInt

// ----------------------------
// Data model & layout tetap lokal di file ini (koordinat TETAP)
// ----------------------------
data class Slot(
    val id: String,
    val xPct: Float,  // 0f..1f dari kiri
    val yPct: Float,  // 0f..1f dari atas
    val wPct: Float,  // 0f..1f dari lebar container
    val hPct: Float,  // 0f..1f dari tinggi container
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
// SCREEN
// ----------------------------
@Composable
fun LiveParkingPage(
    vm: LiveParkingViewModel = viewModel()
) {
    val loading by vm.loading.collectAsStateWithLifecycle()
    val error by vm.error.collectAsStateWithLifecycle()
    val statusById by vm.statusById.collectAsStateWithLifecycle()

    // Base lot dengan KOORDINAT TETAP (ubah sesuai denah kamu)
    val baseLot = remember { sampleLot(R.drawable.liveparkingmap) }

    // Gabungkan status dari VM ke lot (hanya warna yang berubah)
    val coloredLot = remember(baseLot, statusById) {
        val updatedSlots = baseLot.slots.map { s ->
            val status = statusById[s.id]?.lowercase()
            val isOcc = status == "occupied"
            val isAcc = status == "disabled_slot"
            s.copy(
                occupied = isOcc,
                accessible = isAcc
            )
        }
        val used = updatedSlots.count { it.occupied }
        val free = updatedSlots.size - used
        baseLot.copy(slots = updatedSlots, used = used, free = free)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(vertical = 16.dp, horizontal = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Live Parking",
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
        )
        Text(
            "Kantong Parkir FT UGM",
            style = MaterialTheme.typography.bodyMedium
        )
        Spacer(Modifier.height(12.dp))

        when {
            loading -> {
                CircularProgressIndicator()
            }
            error != null -> {
                Text(text = error ?: "-", color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(8.dp))
                Button(onClick = vm::reload) { Text("Coba Lagi") }
            }
            else -> {
                LotCard(
                    lot = coloredLot,
                    onRefresh = vm::reload
                )
            }
        }
    }
}

// ----------------------------
// CARD (pakai koordinat tetap, warna dari status)
// ----------------------------
@Composable
private fun LotCard(
    lot: Lot,
    onRefresh: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(6.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.98f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    lot.name,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
                )
                AssistChip(
                    onClick = { /* optional */ },
                    label = { Text("${lot.free} kosong • ${lot.used} terpakai") }
                )
            }

            Spacer(Modifier.height(10.dp))

            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
                    .clip(RoundedCornerShape(12.dp))
            ) {
                val density = LocalDensity.current // ✅ Tambahkan ini

                val boxWidthPx = with(density) { maxWidth.toPx() }
                val boxHeightPx = with(density) { maxHeight.toPx() }

                Image(
                    painter = painterResource(lot.imageRes),
                    contentDescription = lot.name,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )

                lot.slots.forEach { s ->
                    val xPct = s.xPct.coerceIn(0f, 1f)
                    val yPct = s.yPct.coerceIn(0f, 1f)
                    val wPct = s.wPct.coerceIn(0f, 1f)
                    val hPct = s.hPct.coerceIn(0f, 1f)

                    val wPx = (boxWidthPx * wPct).coerceAtLeast(1f)
                    val hPx = (boxHeightPx * hPct).coerceAtLeast(1f)

                    val maxX = (boxWidthPx - wPx).coerceAtLeast(0f)
                    val maxY = (boxHeightPx - hPx).coerceAtLeast(0f)

                    val xPx = (boxWidthPx * xPct).coerceIn(0f, maxX)
                    val yPx = (boxHeightPx * yPct).coerceIn(0f, maxY)

                    val wDp = with(density) { wPx.toDp() }
                    val hDp = with(density) { hPx.toDp() }

                    val baseColor = when {
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
                        Text(
                            text = s.id,
                            style = MaterialTheme.typography.labelMedium,
                            color = Color.White
                        )
                    }
                }
            }

            Spacer(Modifier.height(12.dp))
            Button(
                onClick = onRefresh,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            ) { Text("Refresh") }
        }
    }
}

// ----------------------------
// KOORDINAT TETAP (sesuaikan sendiri)
// ----------------------------
private fun sampleLot(@DrawableRes mapRes: Int): Lot {
    val slots = listOf(
        Slot("S1", 0.14f, 0.50f, 0.10f, 0.33f),
        Slot("S2", 0.30f, 0.50f, 0.10f, 0.33f),
        Slot("S3", 0.45f, 0.50f, 0.10f, 0.33f),
        Slot("S4", 0.61f, 0.50f, 0.10f, 0.33f),
        Slot("S5", 0.76f, 0.50f, 0.10f, 0.33f),
        // contoh slot aksesibilitas tetap biru kalau status DB = disabled_slot
        // default-nya accessible=false, akan di-set dari VM
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

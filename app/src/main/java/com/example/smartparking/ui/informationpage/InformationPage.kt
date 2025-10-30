package com.example.smartparking.ui.informationpage

import android.content.res.Configuration
import androidx.annotation.DrawableRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.OpenInNew
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun InformationPage() {
    val bg = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.95f),
                Color.White,
                GradientBottom.copy(alpha = 0.95f)
            )
        )
    }
    val uri = LocalUriHandler.current
    val Navy = Color(0xFF0A2342)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .windowInsetsPadding(WindowInsets.systemBars)
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(bottom = 20.dp)
        ) {
            /* ===== Header ala homepage, diberi jarak ekstra ===== */
            item {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 4.dp)
                ) {
                    Image(
                        painter = painterResource(R.drawable.ugm_logo),
                        contentDescription = "UGM Logo",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.size(64.dp)
                    )
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = buildAnnotatedString {
                            append("Informasi ")
                            withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                                append("Safety Car Riding")
                            }
                            append("\n")
                            withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) {
                                append("di Fakultas Teknik UGM")
                            }
                        },
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontSize = 20.sp,
                            lineHeight = 27.sp,
                            textAlign = TextAlign.Center
                        ),
                        textAlign = TextAlign.Center
                    )
                    Spacer(Modifier.height(8.dp))
                }
            }

            /* ===== Section 1: Pengantar (sticker kiri) ===== */
            item {
                ForegroundStickerCard(
                    sticker = R.drawable.info2,
                    stickerSide = StickerSide.START,
                    stickerAlpha = 0.85f,
                    // Tambah padding agar teks tidak menempel stiker & tepi kartu
                    innerPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                    extraContentPadding = PaddingValues(start = 74.dp, end = 8.dp, top = 2.dp, bottom = 2.dp),
                    stickerSize = 56.dp
                ) {
                    Text(
                        "Dalam mendukung penerapan Safety, Health, and Environment (SHE) sesuai Peraturan Rektor UGM, seluruh sivitas Fakultas Teknik wajib mengutamakan keselamatan saat berkendara di lingkungan kampus.",
                        style = MaterialTheme.typography.bodyLarge,
                        lineHeight = 20.sp
                    )
                }
            }

            /* ===== Section 2: Poin aturan (sticker kanan) ===== */
            item {
                ForegroundStickerCard(
                    sticker = R.drawable.info3,
                    stickerSide = StickerSide.END,
                    stickerAlpha = 0.85f,
                    innerPadding = PaddingValues(horizontal = 16.dp, vertical = 18.dp),
                    extraContentPadding = PaddingValues(start = 8.dp, end = 82.dp, top = 2.dp, bottom = 2.dp),
                    stickerSize = 64.dp
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Bullet("Memakai helm (untuk motor) dan sabuk pengaman (untuk mobil)")
                        Bullet("Membatasi kecepatan maksimal 30 km/jam")
                        Bullet("Mematuhi rambu lalu lintas dan portal kendaraan")
                        Bullet("Parkir tertib di lokasi yang telah ditentukan")
                        Bullet("Tidak menggunakan HP saat berkendara, tidak melawan arus, dan tidak menghalangi jalur evakuasi")
                    }
                }
            }

            /* ===== Section 3: Penutup (sticker kanan) ===== */
            item {
                ForegroundStickerCard(
                    sticker = R.drawable.info4,
                    stickerSide = StickerSide.END,
                    stickerAlpha = 0.9f,
                    innerPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                    extraContentPadding = PaddingValues(start = 8.dp, end = 76.dp),
                    stickerSize = 60.dp
                ) {
                    Text(
                        "Keselamatan adalah tanggung jawab bersama. Mari wujudkan lingkungan FT UGM yang aman, sehat, dan tertib ✨",
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                        lineHeight = 20.sp
                    )
                }
            }

            /* ===== CTA navy–putih ===== */
            item {
                Button(
                    onClick = { uri.openUri("https://ugm.id/TegakSHEI") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Navy,
                        contentColor = Color.White
                    )
                ) {
                    Icon(Icons.Outlined.OpenInNew, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Lihat panduan lengkap TegakSHEI")
                }
            }
        }
    }
}

/* ---------- Building blocks ---------- */

private enum class StickerSide { START, END }

@Composable
private fun ForegroundStickerCard(
    @DrawableRes sticker: Int,
    stickerSide: StickerSide,
    stickerAlpha: Float,
    innerPadding: PaddingValues,
    extraContentPadding: PaddingValues,
    stickerSize: Dp,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        elevation = CardDefaults.cardElevation(6.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.98f))
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(innerPadding)
        ) {
            Image(
                painter = painterResource(sticker),
                contentDescription = null,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .align(if (stickerSide == StickerSide.START) Alignment.TopStart else Alignment.CenterEnd)
                    .size(stickerSize)
                    .alpha(stickerAlpha)
            )
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(extraContentPadding),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                content = content
            )
        }
    }
}

@Composable
private fun Bullet(text: String) {
    Row(verticalAlignment = Alignment.Top) {
        Box(
            modifier = Modifier
                .padding(top = 7.dp, end = 10.dp)
                .size(7.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary)
        )
        Text(text, style = MaterialTheme.typography.bodyLarge, lineHeight = 20.sp)
    }
}

/* ---------- Previews ---------- */

@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "Information – Light")
@Composable
private fun PreviewInformationLight() {
    SmartParkingTheme(dynamicColor = false) { InformationPage() }
}

@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_YES, name = "Information – Dark")
@Composable
private fun PreviewInformationDark() {
    SmartParkingTheme(darkTheme = true, dynamicColor = false) { InformationPage() }
}

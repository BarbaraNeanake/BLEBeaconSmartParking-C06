package com.example.smartparking.ui.informationpage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun InformationPage(
    onMenuClick: () -> Unit = {}
) {
    val bg = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.95f),
                Color.White,
                GradientBottom.copy(alpha = 0.95f)
            )
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .windowInsetsPadding(WindowInsets.systemBars)
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp)
        ) {

            // Header
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = onMenuClick) {
                        Icon(Icons.Filled.Menu, contentDescription = "Menu")
                    }
                    Spacer(Modifier.width(8.dp))

                    val title = buildAnnotatedString {
                        append("Informasi ")
                        withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                            append("Safety Car Riding")
                        }
                        append("\n")
                        withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) {
                            append("di Fakultas Teknik UGM")
                        }
                    }

                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontSize = 20.sp,
                            lineHeight = 24.sp
                        ),
                        modifier = Modifier.weight(1f)
                    )

                    // decorative sticker on the right (warning sign)
                    Image(
                        painter = painterResource(R.drawable.info1),
                        contentDescription = null,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .size(42.dp)
                            .padding(start = 6.dp)
                    )
                }
            }

            // Card 1 (pengantar) + sticker left
            item {
                Box(modifier = Modifier.fillMaxWidth()) {
                    // sticker left
                    Image(
                        painter = painterResource(R.drawable.info2),
                        contentDescription = null,
                        modifier = Modifier
                            .size(56.dp)
                            .align(Alignment.TopStart)
                            .offset(x = (-6).dp, y = (-8).dp)
                    )

                    InfoCard {
                        Text(
                            "Dalam mendukung penerapan Safety, Health, and Environment (SHE) sesuai Peraturan Rektor UGM, seluruh sivitas Fakultas Teknik wajib mengutamakan keselamatan saat berkendara di lingkungan kampus.",
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }

            // Card 2 (list aturan) + seatbelt sticker on the right
            item {
                Box(modifier = Modifier.fillMaxWidth()) {
                    InfoCard {
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            Bullet("Memakai helm (untuk motor) dan sabuk pengaman (untuk mobil)")
                            Bullet("Membatasi kecepatan maksimal 30 km/jam")
                            Bullet("Mematuhi rambu lalu lintas dan portal kendaraan")
                            Bullet("Parkir tertib di lokasi yang telah ditentukan")
                            Bullet("Tidak menggunakan HP saat berkendara, tidak melawan arus, dan tidak menghalangi jalur evakuasi")
                        }
                    }

                    Image(
                        painter = painterResource(R.drawable.info3),
                        contentDescription = null,
                        modifier = Modifier
                            .size(84.dp)
                            .align(Alignment.CenterEnd)
                            .offset(x = 4.dp, y = 24.dp)
                    )
                }
            }

            // Card 3 (penutup) + car sticker bottom right + megaphone bottom left
            item {
                Box(modifier = Modifier.fillMaxWidth()) {
                    InfoCard {
                        Text(
                            "Keselamatan adalah tanggung jawab bersama. Mari wujudkan lingkungan FT UGM yang aman, sehat, dan tertib ✨",
                            style = MaterialTheme.typography.bodyLarge.copy(
                                fontWeight = FontWeight.SemiBold
                            )
                        )
                    }

                    Image(
                        painter = painterResource(R.drawable.info4),
                        contentDescription = null,
                        modifier = Modifier
                            .size(86.dp)
                            .align(Alignment.BottomEnd)
                            .offset(y = 6.dp)
                    )
                }
            }

            // bottom spacing (no OS handle bar drawn by us — this is just space)
            item { Spacer(Modifier.height(16.dp)) }
        }
    }
}

/* ---------- Small building blocks ---------- */

@Composable
private fun InfoCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 12.dp, end = 12.dp),
        shape = RoundedCornerShape(10.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
                .copy(alpha = 0.96f)
        )
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            content = content
        )
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
        Text(text, style = MaterialTheme.typography.bodyLarge)
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

package com.example.smartparking.ui.informationpage

import android.content.res.Configuration
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Menu
//import androidx.compose.material.icons.filled.Security
//import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
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
                        withStyle(SpanStyle(fontWeight = FontWeight.Medium)) {
                            append("di Fakultas Teknik UGM")
                        }
                    }
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleLarge.copy(fontSize = 20.sp),
                        modifier = Modifier.weight(1f)
                    )
                    Text("⚠️", fontSize = 22.sp, modifier = Modifier.padding(start = 6.dp))
                }
            }

            item {
                DecoratedCard(
                    leading = { Text("📣", fontSize = 22.sp) }
                ) {
                    Text(
                        "Dalam mendukung penerapan Safety, Health, and Environment (SHE) sesuai Peraturan Rektor UGM, seluruh sivitas Fakultas Teknik wajib mengutamakan keselamatan saat berkendara di lingkungan kampus.",
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }

            item {
                DecoratedCard(
                    leading = { Icon(Icons.Filled.Build, contentDescription = null) }
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Bullet("Memakai helm (untuk motor) dan sabuk pengaman (untuk mobil)")
                        Bullet("Membatasi kecepatan maksimal 30 km/jam")
                        Bullet("Mematuhi rambu lalu lintas dan portal kendaraan")
                        Bullet("Parkir tertib di lokasi yang telah ditentukan")
                        Bullet("Tidak menggunakan HP saat berkendara, tidak melawan arus, dan tidak menghalangi jalur evakuasi")
                    }
                }
            }

            item {
                DecoratedCard(
                    leading = { Text("✅", fontSize = 22.sp) }
                ) {
                    Text(
                        "Keselamatan adalah tanggung jawab bersama. Mari wujudkan lingkungan FT UGM yang aman, sehat, dan tertib ✨",
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold)
                    )
                }
            }

            item {
                Spacer(Modifier.height(8.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    contentAlignment = Alignment.Center // <- jangan pakai CenterHorizontally
                ) {
                    Box(
                        modifier = Modifier
                            .width(120.dp)
                            .height(6.dp)
                            .clip(RoundedCornerShape(50))
                            .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f))
                    )
                }
            }
        }
    }
}

/* ---------- Helpers ---------- */

@Composable
private fun DecoratedCard(
    leading: @Composable (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit
) {
    Row(
        verticalAlignment = Alignment.Top,
        modifier = Modifier.fillMaxWidth()
    ) {
        if (leading != null) {
            Box(
                modifier = Modifier
                    .padding(top = 6.dp, end = 8.dp)
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) { leading() }
        }

        Card(
            modifier = Modifier.weight(1f),
            shape = RoundedCornerShape(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
            )
        ) {
            Column(Modifier.padding(16.dp), content = content)
        }
    }
}

@Composable
private fun Bullet(text: String) {
    Row(verticalAlignment = Alignment.Top) {
        Box(
            modifier = Modifier
                .padding(top = 6.dp, end = 10.dp)
                .size(6.dp)
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

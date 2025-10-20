package com.example.smartparking.ui.landingpage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

// Pilihan warna aman (navy / hitam)
private val SparkNavy = Color(0xFF0A2342)
private val SparkBlack = Color(0xFF111111)

@Composable
fun LandingPageScreen(
    brandName: String = "SPARK",
    subTitle: String = "Smart Parking FT UGM",
    onNavigateNext: () -> Unit = {},
    modifier: Modifier = Modifier,
    brandColor: Color = SparkNavy,
    brandFont: FontFamily = FontFamily.SansSerif,
    subtitleFont: FontFamily = FontFamily.SansSerif,
   // appName: String = "Smart Parking FT UGM"
) {
    val gradient = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.85f),
                Color.White,
                GradientBottom.copy(alpha = 0.85f)
            )
        )
    }

    Surface(modifier = modifier.fillMaxSize(), color = Color.Transparent) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(gradient)
                .clickable(onClick = onNavigateNext)
                .padding(horizontal = 24.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Image(
                    painter = painterResource(id = R.drawable.ugm_logo),
                    contentDescription = "UGM Logo",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.size(96.dp)
                )

                Spacer(Modifier.height(16.dp))

                // --- BRAND: SPARK (navy/black + font berbeda) ---
                Text(
                    text = brandName,
                    textAlign = TextAlign.Center,
                    color = brandColor,
                    style = MaterialTheme.typography.displayLarge.copy(
                        fontSize = 44.sp,
                        fontWeight = FontWeight.ExtraBold,
                        letterSpacing = 1.2.sp,
                        fontFamily = brandFont
                    )
                )

                Spacer(Modifier.height(6.dp))

                // --- Subtitle: Smart Parking FT UGM (font lain) ---
                Text(
                    text = subTitle,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.90f),
                    style = MaterialTheme.typography.headlineSmall.copy(
                        fontSize = 20.sp,
                        fontFamily = subtitleFont,
                        fontWeight = FontWeight.Medium
                    )
                )
            }
        }
    }
}

/* ============================== PREVIEWS ============================== */

@Preview(name = "Landing – Navy", uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
private fun PreviewLandingNavy() {
    SmartParkingTheme(darkTheme = false, dynamicColor = false) {
        LandingPageScreen(brandColor = SparkNavy)
    }
}

@Preview(name = "Landing – Black", uiMode = Configuration.UI_MODE_NIGHT_YES, showBackground = true)
@Composable
private fun PreviewLandingBlack() {
    SmartParkingTheme(darkTheme = true, dynamicColor = false) {
        LandingPageScreen(brandColor = SparkBlack)
    }
}

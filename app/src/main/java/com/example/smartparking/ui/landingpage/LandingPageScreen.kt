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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

/**
 * Landing page: gradasi soft-blue, logo UGM, nama app.
 * Klik di mana pun pada layar -> onNavigateNext()
 */
@Composable
fun LandingPage(
    appName: String = "Smart Parking FT UGM",
    onNavigateNext: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    // Gradasi lembut (atas & bawah biru muda, tengah putih)
    val gradient = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.85f),
                Color.White,
                GradientBottom.copy(alpha = 0.85f)
            )
        )
    }

    Surface(
        modifier = modifier.fillMaxSize(),
        color = Color.Transparent
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(gradient)
                .clickable(onClick = onNavigateNext) // <<-- klik di mana pun
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

                Text(
                    text = appName,
                    style = MaterialTheme.typography.displayLarge.copy(fontSize = 28.sp),
                    color = MaterialTheme.colorScheme.onBackground,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

/* ============================== PREVIEWS ============================== */

@Preview(name = "Landing – Light", uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
private fun PreviewLandingLight() {
    SmartParkingTheme(darkTheme = false, dynamicColor = false) {
        LandingPage()
    }
}

@Preview(name = "Landing – Dark", uiMode = Configuration.UI_MODE_NIGHT_YES, showBackground = true)
@Composable
private fun PreviewLandingDark() {
    SmartParkingTheme(darkTheme = true, dynamicColor = false) {
        LandingPage()
    }
}

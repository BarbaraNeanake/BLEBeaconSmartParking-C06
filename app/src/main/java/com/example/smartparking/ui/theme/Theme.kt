package com.example.smartparking.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.Color


private val LightColorScheme = lightColorScheme(
    primary = BluePrimary,
    onPrimary = OnPrimary,
    secondary = Blue200,
    tertiary = Blue100,

    background = BackgroundLight,
    onBackground = OnBackground,
    surface = SurfaceLight,
    onSurface = OnSurface
)

private val DarkColorScheme = darkColorScheme(
    primary = Blue200,
    onPrimary = OnBackground,
    secondary = Blue100,
    tertiary = Blue100,

    background = Color(0xFF111318),
    onBackground = Color(0xFFE6E9EF),
    surface = Color(0xFF191C20),
    onSurface = Color(0xFFE6E9EF)
)

@Composable
fun SmartParkingTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        // Dynamic color (Android 12+) boleh dipakai untuk mode gelap,
        // tapi untuk feel landing yang konsisten, kita lock ke palette sendiri saat light.
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && darkTheme -> {
            val context = LocalContext.current
            dynamicDarkColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
package com.example.smartparking.ui.theme
import androidx.compose.ui.unit.sp

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp


// Kalau belum punya custom font, pakai default dulu.
// Nanti gampang ganti ke font UGM/Inter/Manrope dengan FontResource.
val AppFont = FontFamily.SansSerif

val Typography = Typography(
    displayLarge = TextStyle(        // untuk judul besar di landing
        fontFamily = AppFont,
        fontWeight = FontWeight.SemiBold,
        fontSize = 36.sp,
        lineHeight = 42.sp
    ),
    headlineMedium = TextStyle(      // subjudul seperti "Fakultas Teknik UGM"
        fontFamily = AppFont,
        fontWeight = FontWeight.Medium,
        fontSize = 20.sp,
        lineHeight = 28.sp
    ),
    titleMedium = TextStyle(         // label kecil / section title
        fontFamily = AppFont,
        fontWeight = FontWeight.Medium,
        fontSize = 16.sp,
        lineHeight = 22.sp
    ),
    bodyLarge = TextStyle(           // paragraf umum
        fontFamily = AppFont,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = AppFont,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp
    ),
    labelLarge = TextStyle(          // tombol
        fontFamily = AppFont,
        fontWeight = FontWeight.SemiBold,
        fontSize = 14.sp,
        lineHeight = 16.sp
    )
)

package com.example.smartparking.ui.theme

import androidx.compose.ui.graphics.Color

// Brand Blues (soft & friendly)
val BluePrimary = Color(0xFF2F80ED)      // aksen utama (buttons, highlight)
val Blue600     = Color(0xFF1C6AE4)      // shade lebih gelap (pressed/active)
val Blue200     = Color(0xFF90CAF9)      // soft blue untuk secondary chips/badges
val Blue100     = Color(0xFFE3F2FD)      // very light blue (tints)

// Neutrals
val Grey900 = Color(0xFF212121)          // title text
val Grey700 = Color(0xFF424242)          // body text
val Grey500 = Color(0xFF9E9E9E)          // secondary text
val Grey100 = Color(0xFFF5F7FA)          // subtle surface

// Backgrounds
val BackgroundLight = Color(0xFFFFFFFF)  // putih bersih untuk base
val SurfaceLight    = Color(0xFFFFFFFF)  // kartu/surface utamanya

// On-colors
val OnPrimary = Color(0xFFFFFFFF)
val OnBackground = Grey900
val OnSurface = Grey900

// (Opsional) warna bantuan untuk gradasi landing (dipakai di Composable, bukan di scheme)
val GradientTop    = Color(0xFFB3E5FC)   // biru muda lembut (atas)
val GradientBottom = Color(0xFF81D4FA)   // biru sedikit lebih kuat (bawah)

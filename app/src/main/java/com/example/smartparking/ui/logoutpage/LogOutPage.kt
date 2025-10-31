package com.example.smartparking.ui.logoutpage

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun LogoutPage(
    vm: LogoutViewModel = viewModel(),
    onCancel: () -> Unit,
    onLoggedOut: () -> Unit
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

    if (ui.showDialog) {
        LogoutDialog(
            loading = ui.loading,
            onCancel = { vm.cancelDialog(); onCancel() },
            onConfirm = { vm.confirmLogout(onSuccess = onLoggedOut) }
        )
    } else {
        // dialog ditutup → balik ke screen sebelumnya
        LaunchedEffect(Unit) { onCancel() }
    }
}

/* ------------ UI Dialog (Hi-Fi) ------------ */
@Composable
private fun LogoutDialog(
    loading: Boolean,
    onCancel: () -> Unit,
    onConfirm: () -> Unit
) {
    Dialog(onDismissRequest = onCancel) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp),
            contentAlignment = Alignment.TopCenter
        ) {
            // Card utama
            Card(
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(8.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 36.dp)
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 18.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Apakah kamu yakin untuk keluar?",
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.SemiBold
                        )
                    )

                    Spacer(Modifier.height(14.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        OutlinedButton(
                            onClick = onCancel,
                            enabled = !loading,
                            modifier = Modifier
                                .weight(1f)
                                .height(44.dp),
                            shape = RoundedCornerShape(14.dp)
                        ) { Text("Batal") }

                        GradientButton(
                            text = if (loading) "..." else "Ya",
                            onClick = onConfirm,
                            modifier = Modifier
                                .weight(1f)
                                .height(44.dp),
                            enabled = !loading,
                            shape = RoundedCornerShape(14.dp)
                        )
                    }
                }
            }

            // Icon bulat di atas card
            Box(
                modifier = Modifier
                    .size(54.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.surface),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Logout, // non-deprecated, auto RTL
                    contentDescription = "Logout Icon",
                    tint = Color(0xFFFF7043),
                    modifier = Modifier.size(32.dp)
                )
            }
        }
    }
}

/* ------------ Util: Tombol Gradien ------------ */
@Composable
private fun GradientButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,                    // modifier dulu (lint happy)
    enabled: Boolean = true,
    shape: RoundedCornerShape = RoundedCornerShape(12.dp)
) {
    val gradient = Brush.horizontalGradient(
        listOf(Color(0xFF9CE5FF), Color(0xFF3DB6FF))
    )
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = shape,
        contentPadding = PaddingValues(),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = Color.White,
            disabledContainerColor = Color.Transparent,
            disabledContentColor = Color.White.copy(alpha = 0.6f)
        ),
        modifier = modifier.background(gradient, shape)
    ) { Text(text) }
}

package com.example.smartparking.ui.editpasspage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.R
import com.example.smartparking.ui.theme.GradientBottom
import com.example.smartparking.ui.theme.GradientTop
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun EditPassPage(
    vm: EditPassViewModel = viewModel(),
    onBackToLogin: () -> Unit = {}
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

    // kalau sukses → balik ke login
    LaunchedEffect(ui.success) {
        if (ui.success) onBackToLogin()
    }

    val bg = remember {
        Brush.verticalGradient(
            listOf(
                GradientTop.copy(alpha = 0.9f),
                Color.White,
                GradientBottom.copy(alpha = 0.9f)
            )
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bg)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // ===== Header (logo + brand) =====
            Spacer(Modifier.height(16.dp))
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.size(56.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "SPARK",
                style = MaterialTheme.typography.headlineSmall.copy(
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 24.sp,
                    color = Color(0xFF0A2342)
                )
            )
            Text(
                text = "Smart Parking System",
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(Modifier.height(18.dp))

            // ===== Card form =====
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(10.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        "Reset Password",
                        style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                        textAlign = TextAlign.Center
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Masukkan email akun dan kata sandi baru.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )

                    Spacer(Modifier.height(16.dp))

                    // Email
                    OutlinedTextField(
                        value = ui.email,
                        onValueChange = vm::onEmailChanged,
                        label = { Text("Email UGM") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(Modifier.height(12.dp))

                    // Password baru
                    OutlinedTextField(
                        value = ui.newPassword,
                        onValueChange = vm::onNewPasswordChanged,
                        label = { Text("Password baru") },
                        singleLine = true,
                        visualTransformation = if (ui.showNew) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            Text(
                                if (ui.showNew) "Hide" else "Show",
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier
                                    .padding(end = 8.dp)
                                    .clickable { vm.toggleShowNew() }
                            )
                        },
                        supportingText = {
                            Text("Min. 6 karakter • campur huruf & angka untuk keamanan.")
                        },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(Modifier.height(12.dp))

                    // Konfirmasi password
                    OutlinedTextField(
                        value = ui.confirmPassword,
                        onValueChange = vm::onConfirmPasswordChanged,
                        label = { Text("Konfirmasi password") },
                        singleLine = true,
                        visualTransformation = if (ui.showConfirm) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            Text(
                                if (ui.showConfirm) "Hide" else "Show",
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier
                                    .padding(end = 8.dp)
                                    .clickable { vm.toggleShowConfirm() }
                            )
                        },
                        modifier = Modifier.fillMaxWidth()
                    )

                    if (ui.errorMessage != null) {
                        Spacer(Modifier.height(10.dp))
                        Text(
                            text = ui.errorMessage!!,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center
                        )
                    }

                    Spacer(Modifier.height(16.dp))

                    // CTA
                    Button(
                        onClick = vm::submitReset,
                        enabled = !ui.loading,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        if (ui.loading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                color = MaterialTheme.colorScheme.onPrimary,
                                strokeWidth = 2.dp
                            )
                        } else {
                            Text("Simpan Password")
                        }
                    }

                    TextButton(
                        onClick = onBackToLogin,
                        modifier = Modifier.align(Alignment.End)
                    ) { Text("Kembali ke Login") }
                }
            }
        }
    }
}

/* ===== Preview ===== */
@Preview(
    name = "EditPass – Light",
    uiMode = Configuration.UI_MODE_NIGHT_NO,
    showBackground = true
)
@Composable
private fun PreviewEditPassLight() {
    SmartParkingTheme {
        EditPassPage(onBackToLogin = {})
    }
}

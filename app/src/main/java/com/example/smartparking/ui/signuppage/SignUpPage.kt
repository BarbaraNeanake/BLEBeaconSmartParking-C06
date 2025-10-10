package com.example.smartparking.ui.signuppage

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
fun SignUpPage(
    vm: SignUpViewModel = viewModel(),
    onRegistered: () -> Unit = {},
    onBackToLogin: () -> Unit = {}
) {
    val ui by vm.ui.collectAsStateWithLifecycle()

    LaunchedEffect(ui.registered) { if (ui.registered) onRegistered() }

    SignUpContent(
        ui = ui,
        onName = vm::onName,
        onEmail = vm::onEmail,
        onPlate = vm::onLicensePlate,
        onCountryCode = vm::onCountryCode,
        onPhone = vm::onPhone,
        onPassword = vm::onPassword,
        onConfirmPassword = vm::onConfirmPassword,
        onTogglePassword = vm::togglePwd,
        onToggleConfirmPassword = vm::toggleConfirmPwd,
        onRegister = vm::register,
        onBackToLogin = onBackToLogin
    )
}

/** ---------- Stateless UI (Preview friendly) ---------- */
@Composable
fun SignUpContent(
    ui: SignUpUiState,
    onName: (String) -> Unit,
    onEmail: (String) -> Unit,
    onPlate: (String) -> Unit,
    onCountryCode: (String) -> Unit,
    onPhone: (String) -> Unit,
    onPassword: (String) -> Unit,
    onConfirmPassword: (String) -> Unit,
    onTogglePassword: () -> Unit,
    onToggleConfirmPassword: () -> Unit,
    onRegister: () -> Unit,
    onBackToLogin: () -> Unit
) {
    // Background gradient yang sama seperti halaman lain
    val gradient = remember {
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
            .background(gradient)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // ===== Header (logo + SPARK + subtitle) — diturunin sedikit =====
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(top = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.size(56.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "SPARK",
                style = MaterialTheme.typography.headlineLarge.copy(
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold
                ),
                color = Color(0xFF0A2342), // navy
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = "Smart Parking System",
                style = MaterialTheme.typography.titleMedium.copy(
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold
                ),
                textAlign = TextAlign.Center
            )
        }

        // ===== Card form (putih + bayangan) =====
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.Center),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(defaultElevation = 10.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Title & link login (center)
                Text(
                    "Sign Up",
                    style = MaterialTheme.typography.headlineLarge.copy(
                        fontWeight = FontWeight.Bold,
                        fontSize = 28.sp
                    ),
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(6.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Already have an account? ")
                    Text(
                        "Login",
                        color = MaterialTheme.colorScheme.primary,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.clickable { onBackToLogin() }
                    )
                }

                Spacer(Modifier.height(16.dp))

                // Full Name
                OutlinedTextField(
                    value = ui.name,
                    onValueChange = onName,
                    label = { Text("Full Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                // Email
                OutlinedTextField(
                    value = ui.email,
                    onValueChange = onEmail,
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                // License Plate (ganti Birth of Date)
                OutlinedTextField(
                    value = ui.licensePlate,
                    onValueChange = onPlate,
                    label = { Text("License Plate (e.g., AB 1234 CD)") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                // Phone (country code + number)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    var expand by remember { mutableStateOf(false) }
                    OutlinedButton(
                        onClick = { expand = true },
                        modifier = Modifier.width(90.dp)
                    ) { Text(ui.countryCode) }
                    DropdownMenu(expanded = expand, onDismissRequest = { expand = false }) {
                        listOf("+62", "+65", "+1", "+81").forEach { code ->
                            DropdownMenuItem(
                                text = { Text(code) },
                                onClick = { onCountryCode(code); expand = false }
                            )
                        }
                    }
                    Spacer(Modifier.width(8.dp))
                    OutlinedTextField(
                        value = ui.phoneNumber,
                        onValueChange = onPhone,
                        label = { Text("Phone Number") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(Modifier.height(12.dp))

                // Password
                OutlinedTextField(
                    value = ui.password,
                    onValueChange = onPassword,
                    label = { Text("Password") },
                    singleLine = true,
                    visualTransformation = if (ui.showPassword) VisualTransformation.None else PasswordVisualTransformation(),
                    trailingIcon = {
                        Text(
                            if (ui.showPassword) "Hide" else "Show",
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clickable { onTogglePassword() }
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                // Confirm Password
                OutlinedTextField(
                    value = ui.confirmPassword,
                    onValueChange = onConfirmPassword,
                    label = { Text("Confirm Password") },
                    singleLine = true,
                    visualTransformation = if (ui.showConfirmPassword) VisualTransformation.None else PasswordVisualTransformation(),
                    trailingIcon = {
                        Text(
                            if (ui.showConfirmPassword) "Hide" else "Show",
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clickable { onToggleConfirmPassword() }
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                )

                if (ui.error != null) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        ui.error!!,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(Modifier.height(16.dp))

                Button(
                    onClick = onRegister,
                    enabled = ui.canSubmit && !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    if (ui.loading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text("Register")
                    }
                }
            }
        }
    }
}

/* -------- PREVIEW -------- */
@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "SignUp – Light")
@Composable
private fun PreviewSignUp() {
    SmartParkingTheme {
        SignUpContent(
            ui = SignUpUiState(
                name = "Barbara Neanake Ajiesti",
                email = "barbaraneanake@ugm.ac.id",
                licensePlate = "AB 1234 CD",
                countryCode = "+62",
                phoneNumber = "81234567890",
                password = "secret123",
                confirmPassword = "secret123",
                canSubmit = true
            ),
            onName = {}, onEmail = {}, onPlate = {},
            onCountryCode = {}, onPhone = {},
            onPassword = {}, onConfirmPassword = {},
            onTogglePassword = {}, onToggleConfirmPassword = {},
            onRegister = {}, onBackToLogin = {}
        )
    }
}

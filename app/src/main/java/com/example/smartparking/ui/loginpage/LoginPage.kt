package com.example.smartparking.ui.loginpage

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
fun LoginPage(
    vm: LoginViewModel = viewModel(),
    onLoginSuccess: () -> Unit = {},
    onSignUpClick: () -> Unit = {},
    onForgotPasswordClick: () -> Unit = {},
    /** Jika true, tombol Log In langsung masuk Home tanpa BE/validasi */
    demoDirectLogin: Boolean = false
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

    // Hanya observe VM kalau TIDAK demo
    LaunchedEffect(ui.isLoggedIn, demoDirectLogin) {
        if (!demoDirectLogin && ui.isLoggedIn) onLoginSuccess()
    }

    LoginContent(
        ui = ui,
        onEmailChange = vm::onEmailChanged,
        onPasswordChange = vm::onPasswordChanged,
        onTogglePassword = vm::togglePasswordVisibility,
        onRememberMeChange = vm::onRememberMeChanged,
        onLoginClick = {
            if (demoDirectLogin) onLoginSuccess() else vm.login()
        },
        onSignUpClick = onSignUpClick,
        onForgotPasswordClick = onForgotPasswordClick,
        showError = !demoDirectLogin
    )
}

/** Pure UI (stateless) */
@Composable
private fun LoginContent(
    ui: LoginUiState,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onTogglePassword: () -> Unit,
    onRememberMeChange: (Boolean) -> Unit,
    onLoginClick: () -> Unit,
    onSignUpClick: () -> Unit,
    onForgotPasswordClick: () -> Unit,
    showError: Boolean = true
) {
    val gradient = remember {
        Brush.verticalGradient(
            listOf(GradientTop.copy(0.9f), Color.White, GradientBottom.copy(0.9f))
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
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
                "SPARK",
                style = MaterialTheme.typography.headlineLarge.copy(
                    fontSize = 28.sp, fontWeight = FontWeight.ExtraBold
                ),
                color = Color(0xFF0A2342),
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(2.dp))
            Text(
                "Smart Parking System",
                style = MaterialTheme.typography.titleMedium.copy(
                    fontSize = 18.sp, fontWeight = FontWeight.SemiBold
                ),
                textAlign = TextAlign.Center
            )
        }

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.Center),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(10.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Login", style = MaterialTheme.typography.headlineLarge.copy(
                    fontWeight = FontWeight.Bold
                ))

                Spacer(Modifier.height(16.dp))

                OutlinedTextField(
                    value = ui.email,
                    onValueChange = onEmailChange,
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                OutlinedTextField(
                    value = ui.password,
                    onValueChange = onPasswordChange,
                    label = { Text("Password") },
                    singleLine = true,
                    visualTransformation = if (ui.passwordVisible)
                        VisualTransformation.None else PasswordVisualTransformation(),
                    trailingIcon = {
                        Text(
                            if (ui.passwordVisible) "Hide" else "Show",
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clickable { onTogglePassword() }
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(checked = ui.rememberMe, onCheckedChange = onRememberMeChange)
                    Text("Remember me")
                    Spacer(Modifier.weight(1f))
                    Text(
                        "Forgot Password ?",
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.clickable { onForgotPasswordClick() }
                    )
                }

                Spacer(Modifier.height(12.dp))

                Button(
                    onClick = onLoginClick,
                    enabled = !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) {
                    if (ui.loading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text("Log In")
                    }
                }

                Spacer(Modifier.height(10.dp))
                OutlinedButton(
                    onClick = onSignUpClick,
                    enabled = !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) { Text("Sign Up") }

                if (showError && ui.errorMessage != null) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        ui.errorMessage,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Preview(showBackground = true, name = "Login – Light", uiMode = Configuration.UI_MODE_NIGHT_NO)
@Composable
private fun PreviewLoginLight() {
    SmartParkingTheme {
        LoginContent(
            ui = LoginUiState(),
            onEmailChange = {},
            onPasswordChange = {},
            onTogglePassword = {},
            onRememberMeChange = {},
            onLoginClick = {},
            onSignUpClick = {},
            onForgotPasswordClick = {}
        )
    }
}

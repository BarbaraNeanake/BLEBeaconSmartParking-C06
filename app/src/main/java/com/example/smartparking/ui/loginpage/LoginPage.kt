package com.example.smartparking.ui.loginpage

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
    onForgotPasswordClick: () -> Unit = {}
) {
    val ui = vm.uiState.collectAsStateWithLifecycle().value

    // Jika login sukses, navigate
    LaunchedEffect(ui.isLoggedIn) {
        if (ui.isLoggedIn) onLoginSuccess()

    }

    LoginContent(
        ui = ui,
        onEmailChange = vm::onEmailChanged,
        onPasswordChange = vm::onPasswordChanged,
        onTogglePassword = vm::togglePasswordVisibility,
        onRememberMeChange = vm::onRememberMeChanged,
        onLoginClick = vm::login,
        onSignUpClick = onSignUpClick,
        onForgotPasswordClick = onForgotPasswordClick
    )
}

@Composable
private fun LoginContent(
    ui: LoginUiState,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onTogglePassword: () -> Unit,
    onRememberMeChange: (Boolean) -> Unit,
    onLoginClick: () -> Unit,
    onSignUpClick: () -> Unit,
    onForgotPasswordClick: () -> Unit
) {
    val gradient = Brush.verticalGradient(
        listOf(
            GradientTop.copy(alpha = 0.9f),
            Color.White,
            GradientBottom.copy(alpha = 0.9f)
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // Header
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(8.dp))
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.size(48.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Smart Parking System",
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontSize = 22.sp,
                    fontWeight = FontWeight.SemiBold
                )
            )
        }

        // Card form
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.Center),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    "Login",
                    style = MaterialTheme.typography.headlineLarge.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Don’t have an account? ")
                    Text(
                        "Sign Up",
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.clickable { onSignUpClick() }
                    )
                }

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
                    visualTransformation = if (ui.passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
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

                Spacer(Modifier.height(10.dp))

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

                when {
                    ui.errorMessage != null -> {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = ui.errorMessage,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center
                        )
                    }
                    ui.isLoggedIn -> {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = "Login berhasil!",
                            color = Color(0xFF2E7D32),
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        }
    }
}

@Preview(
    showBackground = true,
    name = "Login – Light",
    uiMode = Configuration.UI_MODE_NIGHT_NO
)
@Composable
private fun PreviewLoginLight() {
    SmartParkingTheme {
        LoginContent(
            ui = LoginUiState(
                email = "barbara@ugm.ac.id",
                password = "******",
                passwordVisible = false,
                rememberMe = true,
                loading = false,
                errorMessage = null
            ),
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

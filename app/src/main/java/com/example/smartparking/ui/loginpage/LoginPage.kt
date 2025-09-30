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

/**
 * Entry: pakai ViewModel + navigate ketika login sukses.
 */
@Composable
fun LoginPage(
    vm: LoginViewModel = viewModel(),
    onLoginSuccess: () -> Unit = {},
    onSignUpClick: () -> Unit = {},
    onForgotPasswordClick: () -> Unit = {}
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

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
        onGoogleClick = vm::loginWithGoogle,
        onFacebookClick = vm::loginWithFacebook,
        onSignUpClick = onSignUpClick,
        onForgotPasswordClick = onForgotPasswordClick
    )
}

/**
 * Pure UI (stateless) → aman untuk Preview.
 */
@Composable
private fun LoginContent(
    ui: LoginUiState,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onTogglePassword: () -> Unit,
    onRememberMeChange: (Boolean) -> Unit,
    onLoginClick: () -> Unit,
    onGoogleClick: () -> Unit,
    onFacebookClick: () -> Unit,
    onSignUpClick: () -> Unit,
    onForgotPasswordClick: () -> Unit
) {
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

                if (ui.errorMessage != null) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = ui.errorMessage!!,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(Modifier.height(16.dp))

                // divider "Or"
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    HorizontalDivider(Modifier.weight(1f))
                    Text("  Or  ", color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    HorizontalDivider(Modifier.weight(1f))
                }

                Spacer(Modifier.height(12.dp))

                OutlinedButton(
                    onClick = onGoogleClick,
                    enabled = !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) {
                    Text("Continue with Google")
                }

                Spacer(Modifier.height(8.dp))

                OutlinedButton(
                    onClick = onFacebookClick,
                    enabled = !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) {
                    Text("Continue with Facebook")
                }
            }
        }
    }
}

/* ======= PREVIEWS (tidak butuh ViewModel) ======= */

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
            onGoogleClick = {},
            onFacebookClick = {},
            onSignUpClick = {},
            onForgotPasswordClick = {}
        )
    }
}

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
    onForgotPasswordClick: () -> Unit = {}
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

    // auto navigate ketika login sukses
    LaunchedEffect(ui.isLoggedIn) {
        if (ui.isLoggedIn) onLoginSuccess()
    }

    // background gradient
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
        // header (logo + title)
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

        // card form
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
                    onValueChange = vm::onEmailChanged,
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                OutlinedTextField(
                    value = ui.password,
                    onValueChange = vm::onPasswordChanged,
                    label = { Text("Password") },
                    singleLine = true,
                    visualTransformation = if (ui.passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                    trailingIcon = {
                        Text(
                            if (ui.passwordVisible) "Hide" else "Show",
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clickable { vm.togglePasswordVisibility() }
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(checked = ui.rememberMe, onCheckedChange = vm::onRememberMeChanged)
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
                    onClick = { vm.login() },
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
                    onClick = vm::loginWithGoogle,
                    enabled = !ui.loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) {
                    Text("Continue with Google")
                }

                Spacer(Modifier.height(8.dp))

                OutlinedButton(
                    onClick = vm::loginWithFacebook,
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

/* ======= PREVIEW ======= */
@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "Login – Light")
@Composable
private fun PreviewLoginLight() {
    SmartParkingTheme { LoginPage() }
}

annotation class LoginPage

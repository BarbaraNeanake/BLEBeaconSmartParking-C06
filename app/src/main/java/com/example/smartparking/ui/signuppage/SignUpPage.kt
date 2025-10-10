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
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

@OptIn(ExperimentalMaterial3Api::class)
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
        onBirthDatePick = vm::onBirthDateMillis,
        onCountryCode = vm::onCountryCode,
        onPhone = vm::onPhone,
        onPassword = vm::onPassword,
        onTogglePassword = vm::togglePwd,
        onRegister = vm::register,
        onBackToLogin = onBackToLogin
    )
}

/** ---------- Stateless UI so it’s Preview-friendly ---------- */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SignUpContent(
    ui: SignUpUiState,
    onName: (String) -> Unit,
    onEmail: (String) -> Unit,
    onBirthDatePick: (Long) -> Unit,
    onCountryCode: (String) -> Unit,
    onPhone: (String) -> Unit,
    onPassword: (String) -> Unit,
    onTogglePassword: () -> Unit,
    onRegister: () -> Unit,
    onBackToLogin: () -> Unit
) {
    // gradient
    val gradient = remember {
        Brush.verticalGradient(listOf(GradientTop.copy(0.9f), Color.White, GradientBottom.copy(0.9f)))
    }

    val uiDateFormatter = remember { DateTimeFormatter.ofPattern("dd/MM/yyyy") }
    fun toMillisOrNow(dateStr: String): Long {
        return try {
            val ld = LocalDate.parse(dateStr, uiDateFormatter)
            ld.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()
        } catch (_: Exception) {
            Instant.now().toEpochMilli()
        }
    }

    // date picker dialog
    var showDatePicker by remember { mutableStateOf(false) }
    val dateState = rememberDatePickerState(
        initialSelectedDateMillis = toMillisOrNow(ui.birthDateFormatted)
    )
    if (showDatePicker) {
        DatePickerDialog(
            onDismissRequest = { showDatePicker = false },
            confirmButton = {
                TextButton(onClick = {
                    dateState.selectedDateMillis?.let(onBirthDatePick)
                    showDatePicker = false
                }) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = { showDatePicker = false }) { Text("Cancel") } }
        ) { DatePicker(state = dateState) }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // back
        Text(
            text = "←",
            modifier = Modifier
                .align(Alignment.TopStart)
                .clickable { onBackToLogin() }
                .padding(8.dp),
            style = MaterialTheme.typography.titleLarge
        )

        // header logo
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
                modifier = Modifier.size(40.dp)
            )
            Spacer(Modifier.height(8.dp))
        }

        // card
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
                horizontalAlignment = Alignment.Start
            ) {
                Text(
                    "Sign up",
                    style = MaterialTheme.typography.headlineLarge.copy(
                        fontWeight = FontWeight.Bold,
                        fontSize = 28.sp
                    )
                )
                Spacer(Modifier.height(4.dp))
                Row {
                    Text("Already have an account? ")
                    Text(
                        "Login",
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.clickable { onBackToLogin() }
                    )
                }

                Spacer(Modifier.height(16.dp))

                OutlinedTextField(
                    value = ui.name,
                    onValueChange = onName,
                    label = { Text("Full Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                OutlinedTextField(
                    value = ui.email,
                    onValueChange = onEmail,
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(12.dp))

                // Birth date
                OutlinedTextField(
                    value = ui.birthDateFormatted,
                    onValueChange = {},
                    label = { Text("Birth of date") },
                    readOnly = true,
                    trailingIcon = {
                        IconButton(onClick = { showDatePicker = true }) {
                            Icon(
                                painter = painterResource(id = android.R.drawable.ic_menu_today),
                                contentDescription = "Pick date"
                            )
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { showDatePicker = true }
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

                OutlinedTextField(
                    value = ui.password,
                    onValueChange = onPassword,
                    label = { Text("Set Password") },
                    singleLine = true,
                    visualTransformation =
                        if (ui.showPassword) VisualTransformation.None else PasswordVisualTransformation(),
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

                if (ui.error != null) {
                    Spacer(Modifier.height(8.dp))
                    Text(ui.error!!, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Start)
                }

                Spacer(Modifier.height(16.dp))

                Button(
                    onClick = onRegister,
                    enabled = !ui.loading,
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

/* -------- PREVIEW (stateless) -------- */
//@Preview(showBackground = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "SignUp – Light")
//@Composable
//private fun PreviewSignUp() {
//    SmartParkingTheme {
//        SignUpContent(
//            ui = SignUpUiState(
//                name = "Barbara Neanake Ajiesti",
//                email = "barbaraneanake@ugm.ac.id",
//                birthDate = System.currentTimeMillis(),
//                countryCode = "+62",
//                phoneNumber = "81234567890"
//            ),
//            onName = {}, onEmail = {}, onBirthDatePick = {},
//            onCountryCode = {}, onPhone = {},
//            onPassword = {}, onTogglePassword = {},
//            onRegister = {}, onBackToLogin = {}
//        )
//    }
//}

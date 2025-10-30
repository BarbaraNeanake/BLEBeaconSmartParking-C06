package com.example.smartparking.ui.logoutpage

import android.content.res.Configuration
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.ui.theme.SmartParkingTheme

@Composable
fun LogoutPage(
    vm: LogoutViewModel = viewModel(),
    onCancel: () -> Unit,
    onLoggedOut: () -> Unit
) {
    val state by vm.uiState.collectAsState()

    if (state.showDialog) {
        AlertDialog(
            onDismissRequest = { vm.cancelDialog(); onCancel() },
            title = { Text("Log out?") },
            text = {
                Text(
                    if (state.loading) "Signing you out…"
                    else "Anda akan keluar dari akun ini dan kembali ke halaman Login."
                )
            },
            confirmButton = {
                TextButton(
                    enabled = !state.loading,
                    onClick = { vm.confirmLogout(onSuccess = onLoggedOut) }
                ) {
                    Text("Log out")
                }
            },
            dismissButton = {
                TextButton(
                    enabled = !state.loading,
                    onClick = { vm.cancelDialog(); onCancel() }
                ) {
                    Text("Cancel")
                }
            }
        )
    } else {
        // kalau dialog sudah ditutup (Cancel), langsung balik
        LaunchedEffect(Unit) { onCancel() }
    }
}

/* ---- Preview (dummy) ---- */
@Preview(uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
private fun PreviewLogout() {
    SmartParkingTheme {
        LogoutPage(
            onCancel = {},
            onLoggedOut = {}
        )
    }
}

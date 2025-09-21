package com.example.smartparking.ui.home

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@Composable
fun HomeScreen(
    state: HomeUiState,
    onLogoutClick: () -> Unit,
    modifier: Modifier = Modifier // <- added
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(text = "Welcome, ${state.username}!")
            Button(onClick = onLogoutClick) {
                Text(text = "Logout")
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HomeScreenPreview() {
    HomeScreen(
        state = HomeUiState(username = "Preview User"),
        onLogoutClick = {},
        modifier = Modifier.fillMaxSize()
    )
}

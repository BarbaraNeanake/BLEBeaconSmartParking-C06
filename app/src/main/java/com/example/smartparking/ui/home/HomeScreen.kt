package com.example.smartparking.ui.home

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.smartparking.ui.loginpage.LoginViewModel

@Composable
fun HomeScreen(
    vm: LoginViewModel = viewModel() // <- added
) {
    val ui by vm.uiState.collectAsStateWithLifecycle()

    Column {
        Text("Welcome, ${ui.email}")
    }
}

//@Preview(showBackground = true)
//@Composable
//fun HomeScreenPreview() {
//    HomeScreen(
//        state = HomeUiState(nama = HomeViewModel.Load(1)),
//        onLogoutClick = {},
//        modifier = Modifier.fillMaxSize()
//    )
//}

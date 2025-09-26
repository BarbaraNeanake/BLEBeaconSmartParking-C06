package com.example.smartparking

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.navigation.NavController
import androidx.navigation.compose.rememberNavController
import com.example.smartparking.ui.beacontest.BeaconScreen
import com.example.smartparking.ui.home.HomeScreen
//import com.example.smartparking.ui.home.HomeScreenPreview
import com.example.smartparking.ui.home.HomeViewModel
import com.example.smartparking.ui.login.LoginScreen
import com.example.smartparking.ui.login.LoginViewModel
import com.example.smartparking.ui.theme.SmartParkingTheme

class MainActivity : ComponentActivity() {

    // Attach ViewModel to the activity
    private val homeViewModel: HomeViewModel by viewModels()
    private val loginViewModel: LoginViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SmartParkingTheme {
                val state by homeViewModel.uiState.collectAsState()
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
//                    HomeScreen(
//                        viewModel = homeViewModel,
//                        modifier = Modifier.padding(innerPadding)
//                    )
                    BeaconScreen()
//                    LoginScreen(navController, loginViewModel)
                }

            }
        }
    }
}

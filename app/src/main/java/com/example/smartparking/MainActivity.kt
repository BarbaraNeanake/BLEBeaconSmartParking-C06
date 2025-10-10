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
import com.example.smartparking.data.remote.RetrofitClient
import com.example.smartparking.data.repository.ParkingRepository
import com.example.smartparking.ui.beacontest.BeaconScreen
import com.example.smartparking.ui.beacontest.BeaconViewModel
import com.example.smartparking.ui.beacontest.BeaconViewModelFactory
import com.example.smartparking.ui.historypage.HistoryPage
import com.example.smartparking.ui.home.HomeScreen
//import com.example.smartparking.ui.home.HomeScreenPreview
import com.example.smartparking.ui.home.HomeViewModel
//import com.example.smartparking.ui.login.LoginScreen
//import com.example.smartparking.ui.login.LoginViewModel
import com.example.smartparking.ui.theme.SmartParkingTheme
import com.example.smartparking.ui.home.HomeScreen
import com.example.smartparking.ui.homepage.HomePage
import com.example.smartparking.ui.informationpage.InformationPage
import com.example.smartparking.ui.landingpage.LandingPage
import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
import com.example.smartparking.ui.loginpage.LoginPage
import com.example.smartparking.ui.logoutpage.LogoutPage
import com.example.smartparking.ui.navigation.NavGraph
import com.example.smartparking.ui.signuppage.SignUpPage


class MainActivity : ComponentActivity() {

    // Attach ViewModel to the activity
    private val homeViewModel: HomeViewModel by viewModels()
//    private val loginViewModel: LoginViewModel by viewModels()
    private val beaconViewModel: BeaconViewModel by viewModels {
        BeaconViewModelFactory(ParkingRepository(RetrofitClient.parkingApi))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SmartParkingTheme {
                val state by homeViewModel.uiState.collectAsState()
//                NavGraph()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
//                    HomeScreen(
//                        state = state,
//                        onLogoutClick = { /* TODO */ },
//                        modifier = Modifier.padding(innerPadding)
//                    )

                    SignUpPage()
                }

            }
        }
    }
}

package com.example.smartparking

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.smartparking.ui.components.DrawerContent
import com.example.smartparking.ui.editpasspage.EditPassPage
import com.example.smartparking.ui.historypage.HistoryPage
import com.example.smartparking.ui.homepage.HomePage
import com.example.smartparking.ui.informationpage.InformationPage
import com.example.smartparking.ui.landingpage.LandingPageScreen
import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
import com.example.smartparking.ui.loginpage.LoginPage
import com.example.smartparking.ui.logoutpage.LogoutPage
import com.example.smartparking.ui.signuppage.SignUpPage
import com.example.smartparking.ui.theme.SmartParkingTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/* ========================== Routes ========================== */
sealed class Screen(val route: String, val label: String) {
    data object Landing  : Screen("landing", "Landing")
    data object Login    : Screen("login", "Login")
    data object SignUp   : Screen("signup", "Sign Up")
    data object EditPass : Screen("edit_pass", "Reset Password")
    data object Home     : Screen("home", "Home")
    data object Live     : Screen("live_parking", "Live Parking")
    data object History  : Screen("history", "History")
    data object Info     : Screen("information", "Information")
    data object Logout   : Screen("logout", "Logout")
}

/* ===== Drawer wrapper: dipakai cuma di halaman private ===== */
@Composable
private fun WithDrawer(
    selectedRoute: String,
    onNavigateRoute: (String) -> Unit,
    content: @Composable (openDrawer: () -> Unit) -> Unit
) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = true,
        drawerContent = {
            DrawerContent(
                selectedRoute = selectedRoute,
                onItemClick = { route ->
                    // tutup dulu biar nggak glitch, lalu navigate
                    scope.launch {
                        drawerState.close()
                        delay(120)
                        onNavigateRoute(route)
                    }
                }
            )
        }
    ) {
        Box(Modifier.fillMaxSize()) {
            content { scope.launch { drawerState.open() } }

            // Tombol hamburger melayang (kiri-atas)
            Surface(
                tonalElevation = 3.dp,
                shadowElevation = 6.dp,
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.65f),
                shape = MaterialTheme.shapes.extraLarge,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(start = 12.dp, top = 12.dp)
            ) {
                IconButton(onClick = { scope.launch { drawerState.open() } }) {
                    Icon(
                        imageVector = Icons.Filled.Menu,
                        contentDescription = "Menu",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

/* ========================== MainActivity ========================== */
@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            SmartParkingTheme {
                val navController = rememberNavController()

                // full-bleed, no white margins
                Scaffold(
                    containerColor = Color.Transparent,
                    contentWindowInsets = WindowInsets(0)
                ) { innerPadding ->
                    Box(
                        Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    ) {
                        NavHost(
                            navController = navController,
                            startDestination = Screen.Landing.route,
                            modifier = Modifier.fillMaxSize()
                        ) {

                            /* ---------- Public screens (tanpa drawer) ---------- */
                            composable(Screen.Landing.route) {
                                LandingPageScreen(
                                    brandName = "SPARK",
                                    subTitle = "Smart Parking FT UGM",
                                    brandColor = Color(0xFF0A2342),
                                    modifier = Modifier.fillMaxSize(),
                                    onNavigateNext = {
                                        navController.navigate(Screen.Login.route)
                                    }
                                )
                            }

                            composable(Screen.Login.route) {
                                LoginPage(
                                    demoDirectLogin = true,           // ← bypass BE: langsung Home
                                    onLoginSuccess = {
                                        navController.navigate(Screen.Home.route) {
                                            popUpTo(Screen.Landing.route) { inclusive = true } // bersihkan stack
                                            launchSingleTop = true
                                        }
                                    },
                                    onSignUpClick = { navController.navigate(Screen.SignUp.route) },
                                    onForgotPasswordClick = { navController.navigate(Screen.EditPass.route) }
                                )
                            }

                            composable(Screen.SignUp.route) {
                                SignUpPage(
                                    onRegistered = {
                                        navController.navigate(Screen.Home.route) {
                                            popUpTo(Screen.Landing.route) { inclusive = true }
                                            launchSingleTop = true
                                        }
                                    },
                                    onBackToLogin = { navController.popBackStack() }
                                )
                            }

                            composable(Screen.EditPass.route) {
                                EditPassPage(onBackToLogin = { navController.popBackStack() })
                            }

                            /* ---------- Private screens (pakai drawer) ---------- */
                            composable(Screen.Home.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Home.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    }
                                ) { openDrawer ->
                                    HomePage(onMenuClick = openDrawer)
                                }
                            }

                            composable(Screen.Live.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Live.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    }
                                ) { _ ->
                                    LiveParkingPage()
                                }
                            }

                            composable(Screen.History.route) {
                                WithDrawer(
                                    selectedRoute = Screen.History.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    }
                                ) { _ ->
                                    HistoryPage()
                                }
                            }

                            composable(Screen.Info.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Info.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    }
                                ) { _ ->
                                    InformationPage()
                                }
                            }

                            composable(Screen.Logout.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Logout.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    }
                                ) { _ ->
                                    LogoutPage(
                                        onCancel = { navController.popBackStack() },
                                        onLoggedOut = {
                                            navController.navigate(Screen.Login.route) {
                                                popUpTo(Screen.Landing.route) { inclusive = true }
                                                launchSingleTop = true
                                            }
                                        }
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

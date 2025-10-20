package com.example.smartparking

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Map
import androidx.compose.material.icons.outlined.PowerSettingsNew
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
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
private val drawerScreens = listOf(Screen.Home, Screen.Live, Screen.History, Screen.Info, Screen.Logout)

/* ========================== MainActivity ========================== */
@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            SmartParkingTheme {
                val navController = rememberNavController()
                val backstack by navController.currentBackStackEntryAsState()
                val currentRoute = backstack?.destination?.route
                val showDrawer = currentRoute in drawerScreens.map { it.route }

                val drawerState = rememberDrawerState(DrawerValue.Closed)
                val scope = rememberCoroutineScope()

                // --- Drawer yang lebih elegan ---
                ModalNavigationDrawer(
                    drawerState = drawerState,
                    gesturesEnabled = showDrawer,
                    drawerContent = {
                        if (showDrawer) {
                            ModalDrawerSheet(
                                drawerContainerColor = MaterialTheme.colorScheme.surface,
                                drawerTonalElevation = 8.dp
                            ) {
                                // Header
                                Text(
                                    "SPARK",
                                    style = MaterialTheme.typography.headlineSmall,
                                    modifier = Modifier
                                        .background(MaterialTheme.colorScheme.surfaceColorAtElevation(2.dp))
                                        .padding(horizontal = 16.dp, vertical = 20.dp)
                                )
                                NavigationDrawerItem(
                                    label = { Text("Home") },
                                    selected = currentRoute == Screen.Home.route,
                                    onClick = {
                                        scope.launch { drawerState.close() }
                                        navController.navigate(Screen.Home.route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    icon = { Icon(Icons.Outlined.Home, null) }
                                )
                                NavigationDrawerItem(
                                    label = { Text("Live Parking") },
                                    selected = currentRoute == Screen.Live.route,
                                    onClick = {
                                        scope.launch { drawerState.close() }
                                        navController.navigate(Screen.Live.route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    icon = { Icon(Icons.Outlined.Map, null) }
                                )
                                NavigationDrawerItem(
                                    label = { Text("History") },
                                    selected = currentRoute == Screen.History.route,
                                    onClick = {
                                        scope.launch { drawerState.close() }
                                        navController.navigate(Screen.History.route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    icon = { Icon(Icons.Outlined.History, null) }
                                )
                                NavigationDrawerItem(
                                    label = { Text("Information") },
                                    selected = currentRoute == Screen.Info.route,
                                    onClick = {
                                        scope.launch { drawerState.close() }
                                        navController.navigate(Screen.Info.route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    icon = { Icon(Icons.Outlined.Info, null) }
                                )
                                NavigationDrawerItem(
                                    label = { Text("Logout") },
                                    selected = currentRoute == Screen.Logout.route,
                                    onClick = {
                                        scope.launch { drawerState.close() }
                                        navController.navigate(Screen.Logout.route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    icon = { Icon(Icons.Outlined.PowerSettingsNew, null) }
                                )
                            }
                        }
                    }
                ) {
                    // Hilangkan semua insets bawaan supaya full-bleed beneran
                    Scaffold(
                        containerColor = Color.Transparent,
                        contentWindowInsets = WindowInsets(0),
                        topBar = {
                            if (showDrawer) {
                                TopAppBar(
                                    title = { Text(drawerScreens.find { it.route == currentRoute }?.label ?: "") },
                                    navigationIcon = {
                                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                                            Icon(Icons.Filled.Menu, contentDescription = "Menu")
                                        }
                                    }
                                )
                            }
                        }
                    ) { innerPadding ->
                        NavHost(
                            navController = navController,
                            startDestination = Screen.Landing.route,
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(innerPadding) // padding dari TopAppBar saat drawer pages
                        ) {
                            /* ---- Landing: full screen, klik lanjut ke Login ---- */
                            composable(Screen.Landing.route) {
                                LandingPageScreen(
                                    brandName = "SPARK",
                                    subTitle = "Smart Parking FT UGM",
                                    brandColor = Color(0xFF0A2342),
                                    modifier = Modifier.fillMaxSize(),
                                    onNavigateNext = { navController.navigate(Screen.Login.route) }
                                )
                            }

                            /* ---- Login: tombol login → langsung ke Home (tanpa BE) ---- */
                            composable(Screen.Login.route) {
                                // Tidak ada PageContainer/padding global → fullscreen
                                LoginPage(
                                    onLoginSuccess = {
                                        // override: langsung ke Home
                                        navController.navigate(Screen.Home.route) {
                                            popUpTo(Screen.Landing.route) { inclusive = true }
                                            launchSingleTop = true
                                        }
                                    },
                                    onSignUpClick = { navController.navigate(Screen.SignUp.route) },
                                    onForgotPasswordClick = { navController.navigate(Screen.EditPass.route) }
                                )
                            }

                            /* ---- Sign Up (tetap normal) ---- */
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

                            /* ---- Edit/Forgot Password (normal) ---- */
                            composable(Screen.EditPass.route) {
                                EditPassPage(onBackToLogin = { navController.popBackStack() })
                            }

                            /* ---- Private pages (dengan TopAppBar + Drawer) ---- */
                            composable(Screen.Home.route) {
                                HomePage(
                                    onMenuClick = { scope.launch { drawerState.open() } }
                                )
                            }
                            composable(Screen.Live.route) { LiveParkingPage() }
                            composable(Screen.History.route) { HistoryPage() }
                            composable(Screen.Info.route) { InformationPage() }
                            composable(Screen.Logout.route) {
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

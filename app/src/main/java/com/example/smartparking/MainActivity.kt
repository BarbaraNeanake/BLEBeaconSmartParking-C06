package com.example.smartparking

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.Dp
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

/* ======= UI metrics (padding adaptif pakai ukuran layar) ======= */
@Stable
data class UiMetrics(
    val horizontalPadding: Dp,
    val verticalPadding: Dp
)
@Composable
private fun rememberUiMetrics(): UiMetrics {
    val cfg = LocalConfiguration.current
    val w = cfg.screenWidthDp
    val h = cfg.screenHeightDp
    // padding horizontal ~5–6% lebar layar, vertical ~2–3% tinggi layar
    val hp = (w * 0.06f).dp.coerceIn(12.dp, 28.dp)
    val vp = (h * 0.025f).dp.coerceIn(8.dp, 24.dp)
    return remember(w, h) { UiMetrics(horizontalPadding = hp, verticalPadding = vp) }
}

/* ======= Pembungkus halaman agar padding konsisten ======= */
@Composable
private fun PageContainer(pv: PaddingValues, hp: Dp, vp: Dp, content: @Composable () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(pv) // padding dari Scaffold (top app bar, dsb.)
            .padding(horizontal = hp, vertical = vp) // padding global adaptif
    ) { content() }
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
                val backstack by navController.currentBackStackEntryAsState()
                val currentRoute = backstack?.destination?.route
                val showDrawer = currentRoute in drawerScreens.map { it.route }

                val drawerState = rememberDrawerState(DrawerValue.Closed)
                val scope = rememberCoroutineScope()

                val metrics = rememberUiMetrics()

                ModalNavigationDrawer(
                    drawerState = drawerState,
                    gesturesEnabled = showDrawer,
                    drawerContent = {
                        if (showDrawer) {
                            ModalDrawerSheet {
                                Text(
                                    "SPARK",
                                    style = MaterialTheme.typography.titleLarge,
                                    modifier = Modifier.padding(16.dp)
                                )
                                drawerScreens.forEach { scr ->
                                    NavigationDrawerItem(
                                        label = { Text(scr.label) },
                                        selected = currentRoute == scr.route,
                                        onClick = {
                                            scope.launch { drawerState.close() }
                                            navController.navigate(scr.route) {
                                                launchSingleTop = true
                                                // saat pindah antar menu, stack tetap bersih
                                                popUpTo(Screen.Home.route) { inclusive = false }
                                            }
                                        }
                                    )
                                }
                            }
                        }
                    }
                ) {
                    Scaffold(
                        topBar = {
                            if (showDrawer) {
                                TopAppBar(
                                    title = {
                                        Text(drawerScreens.find { it.route == currentRoute }?.label ?: "")
                                    },
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
                            modifier = Modifier.padding(0.dp) // biar PageContainer yang atur
                        ) {
                            /* ---- Auth & intro (tanpa drawer) ---- */
                            composable(Screen.Landing.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    LandingPageScreen(
                                        brandName = "SPARK",
                                        subTitle = "Smart Parking FT UGM",
                                        onNavigateNext = { navController.navigate(Screen.Login.route) },
                                        modifier = TODO(),
                                        brandColor = TODO(),
                                        brandFont = TODO(),
                                        subtitleFont = TODO(),
                                        appName = TODO()
                                    )
                                }
                            }
                            composable(Screen.Login.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    LoginPage(
                                        onLoginSuccess = {
                                            navController.navigate(Screen.Home.route) {
                                                popUpTo(Screen.Landing.route) { inclusive = true }
                                                launchSingleTop = true
                                            }
                                        },
                                        onSignUpClick = { navController.navigate(Screen.SignUp.route) },
                                        onForgotPasswordClick = { navController.navigate(Screen.EditPass.route) }
                                    )
                                }
                            }
                            composable(Screen.SignUp.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
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
                            }
                            composable(Screen.EditPass.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    EditPassPage(onBackToLogin = { navController.popBackStack() })
                                }
                            }

                            /* ---- Private pages (dengan drawer+topbar) ---- */
                            composable(Screen.Home.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    HomePage()
                                }
                            }
                            composable(Screen.Live.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    LiveParkingPage()
                                }
                            }
                            composable(Screen.History.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    HistoryPage()
                                }
                            }
                            composable(Screen.Info.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
                                    InformationPage()
                                }
                            }
                            composable(Screen.Logout.route) {
                                PageContainer(innerPadding, metrics.horizontalPadding, metrics.verticalPadding) {
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

//package com.example.smartparking
//
//import android.os.Bundle
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.enableEdgeToEdge
//import androidx.compose.foundation.background
//import androidx.compose.foundation.layout.WindowInsets
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.padding
//import androidx.compose.material.icons.Icons
//import androidx.compose.material.icons.outlined.History
//import androidx.compose.material.icons.outlined.Home
//import androidx.compose.material.icons.outlined.Info
//import androidx.compose.material.icons.outlined.Map
//import androidx.compose.material.icons.outlined.PowerSettingsNew
//import androidx.compose.material.icons.filled.Menu
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.unit.dp
//import androidx.navigation.compose.NavHost
//import androidx.navigation.compose.composable
//import androidx.navigation.compose.currentBackStackEntryAsState
//import androidx.navigation.compose.rememberNavController
//import com.example.smartparking.ui.editpasspage.EditPassPage
//import com.example.smartparking.ui.historypage.HistoryPage
//import com.example.smartparking.ui.homepage.HomePage
//import com.example.smartparking.ui.informationpage.InformationPage
//import com.example.smartparking.ui.landingpage.LandingPageScreen
//import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
//import com.example.smartparking.ui.loginpage.LoginPage
//import com.example.smartparking.ui.logoutpage.LogoutPage
//import androidx.lifecycle.viewmodel.compose.viewModel
//import androidx.room.Room
//import com.example.smartparking.data.repository.UserRepository
//import com.example.smartparking.data.repository.db.AppDatabase
//import com.example.smartparking.data.network.TokenProvider
//import com.example.smartparking.ui.loginpage.LoginViewModel
//import com.example.smartparking.ui.loginpage.LoginVMFactory
//import androidx.lifecycle.compose.collectAsStateWithLifecycle
//import com.example.smartparking.ui.signuppage.SignUpPage
//import com.example.smartparking.ui.theme.SmartParkingTheme
//import kotlinx.coroutines.launch
//
///* ========================== Routes ========================== */
//sealed class Screen(val route: String, val label: String) {
//    data object Landing  : Screen("landing", "Landing")
//    data object Login    : Screen("login", "Login")
//    data object SignUp   : Screen("signup", "Sign Up")
//    data object EditPass : Screen("edit_pass", "Reset Password")
//    data object Home     : Screen("home", "Home")
//    data object Live     : Screen("live_parking", "Live Parking")
//    data object History  : Screen("history", "History")
//    data object Info     : Screen("information", "Information")
//    data object Logout   : Screen("logout", "Logout")
//}
//private val drawerScreens = listOf(Screen.Home, Screen.Live, Screen.History, Screen.Info, Screen.Logout)
//
///* ========================== MainActivity ========================== */
//@OptIn(ExperimentalMaterial3Api::class)
//class MainActivity : ComponentActivity() {
//    @OptIn(ExperimentalMaterial3Api::class)
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        enableEdgeToEdge()
//
//        setContent {
//            SmartParkingTheme {
//                val navController = rememberNavController()
//                val backstack by navController.currentBackStackEntryAsState()
//                val currentRoute = backstack?.destination?.route
//                val showDrawer = currentRoute in drawerScreens.map { it.route }
//
//                val drawerState = rememberDrawerState(DrawerValue.Closed)
//                val scope = rememberCoroutineScope()
//
//                val appCtx = applicationContext
//
//                val db = remember {
//                    Room.databaseBuilder(appCtx, AppDatabase::class.java, "smartparking.db").build()
//                }
//                val repo = remember { com.example.smartparking.data.repository.UserRepository(db.sessionDao()) }
//
//                val sessionFlow = remember { db.sessionDao().observeSession() }
//                val session by sessionFlow.collectAsStateWithLifecycle(initialValue = null)
//
//                LaunchedEffect(Unit) {
//                    TokenProvider.init(db.sessionDao())
//                }
//
//
//                val startDest = if (session == null) Screen.Landing.route else Screen.Home.route
//
//
//
//                // --- Drawer yang lebih elegan ---
//                ModalNavigationDrawer(
//                    drawerState = drawerState,
//                    gesturesEnabled = showDrawer,
//                    drawerContent = {
//                        if (showDrawer) {
//                            ModalDrawerSheet(
//                                drawerContainerColor = MaterialTheme.colorScheme.surface,
//                                drawerTonalElevation = 8.dp
//                            ) {
//                                // Header
//                                Text(
//                                    "SPARK",
//                                    style = MaterialTheme.typography.headlineSmall,
//                                    modifier = Modifier
//                                        .background(MaterialTheme.colorScheme.surfaceColorAtElevation(2.dp))
//                                        .padding(horizontal = 16.dp, vertical = 20.dp)
//                                )
//                                NavigationDrawerItem(
//                                    label = { Text("Home") },
//                                    selected = currentRoute == Screen.Home.route,
//                                    onClick = {
//                                        scope.launch { drawerState.close() }
//                                        navController.navigate(Screen.Home.route) {
//                                            launchSingleTop = true
//                                            popUpTo(Screen.Home.route) { inclusive = false }
//                                        }
//                                    },
//                                    icon = { Icon(Icons.Outlined.Home, null) }
//                                )
//                                NavigationDrawerItem(
//                                    label = { Text("Live Parking") },
//                                    selected = currentRoute == Screen.Live.route,
//                                    onClick = {
//                                        scope.launch { drawerState.close() }
//                                        navController.navigate(Screen.Live.route) {
//                                            launchSingleTop = true
//                                            popUpTo(Screen.Home.route) { inclusive = false }
//                                        }
//                                    },
//                                    icon = { Icon(Icons.Outlined.Map, null) }
//                                )
//                                NavigationDrawerItem(
//                                    label = { Text("History") },
//                                    selected = currentRoute == Screen.History.route,
//                                    onClick = {
//                                        scope.launch { drawerState.close() }
//                                        navController.navigate(Screen.History.route) {
//                                            launchSingleTop = true
//                                            popUpTo(Screen.Home.route) { inclusive = false }
//                                        }
//                                    },
//                                    icon = { Icon(Icons.Outlined.History, null) }
//                                )
//                                NavigationDrawerItem(
//                                    label = { Text("Information") },
//                                    selected = currentRoute == Screen.Info.route,
//                                    onClick = {
//                                        scope.launch { drawerState.close() }
//                                        navController.navigate(Screen.Info.route) {
//                                            launchSingleTop = true
//                                            popUpTo(Screen.Home.route) { inclusive = false }
//                                        }
//                                    },
//                                    icon = { Icon(Icons.Outlined.Info, null) }
//                                )
//                                NavigationDrawerItem(
//                                    label = { Text("Logout") },
//                                    selected = currentRoute == Screen.Logout.route,
//                                    onClick = {
//                                        scope.launch { drawerState.close() }
//                                        navController.navigate(Screen.Logout.route) {
//                                            launchSingleTop = true
//                                            popUpTo(Screen.Home.route) { inclusive = false }
//                                        }
//                                    },
//                                    icon = { Icon(Icons.Outlined.PowerSettingsNew, null) }
//                                )
//                            }
//                        }
//                    }
//                ) {
//                    // Hilangkan semua insets bawaan supaya full-bleed beneran
//                    Scaffold(
//                        containerColor = Color.Transparent,
//                        contentWindowInsets = WindowInsets(0),
//                        topBar = {
//                            if (showDrawer) {
//                                TopAppBar(
//                                    title = { Text(drawerScreens.find { it.route == currentRoute }?.label ?: "") },
//                                    navigationIcon = {
//                                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
//                                            Icon(Icons.Filled.Menu, contentDescription = "Menu")
//                                        }
//                                    }
//                                )
//                            }
//                        }
//                    ) { innerPadding ->
//                        NavHost(
//                            navController = navController,
//                            startDestination = startDest,
//                            modifier = Modifier
//                                .fillMaxSize()
//                                .padding(innerPadding) // padding dari TopAppBar saat drawer pages
//                        ) {
//                            /* ---- Landing: full screen, klik lanjut ke Login ---- */
//                            composable(Screen.Landing.route) {
//                                LandingPageScreen(
//                                    brandName = "SPARK",
//                                    subTitle = "Smart Parking FT UGM",
//                                    brandColor = Color(0xFF0A2342),
//                                    modifier = Modifier.fillMaxSize(),
//                                    onNavigateNext = { navController.navigate(Screen.Login.route) }
//                                )
//                            }
//
//                            /* ---- Login: tombol login → langsung ke Home (tanpa BE) ---- */
//                            composable(Screen.Login.route) {
//                                val vm: LoginViewModel =
//                                    viewModel(
//                                        factory = LoginVMFactory(repo)
//                                    )
//
//                                LoginPage(
//                                    vm = vm,
//                                    onLoginSuccess = {
//                                        navController.navigate(Screen.Home.route) {
//                                            popUpTo(Screen.Landing.route) { inclusive = true }
//                                            launchSingleTop = true
//                                        }
//                                    },
//                                    onSignUpClick = { navController.navigate(Screen.SignUp.route) },
//                                    onForgotPasswordClick = { navController.navigate(Screen.EditPass.route) }
//                                )
//                            }
//
//
//                            /* ---- Sign Up (tetap normal) ---- */
//                            composable(Screen.SignUp.route) {
//                                SignUpPage(
//                                    onRegistered = {
//                                        navController.navigate(Screen.Home.route) {
//                                            popUpTo(Screen.Landing.route) { inclusive = true }
//                                            launchSingleTop = true
//                                        }
//                                    },
//                                    onBackToLogin = { navController.popBackStack() }
//                                )
//                            }
//
//                            /* ---- Edit/Forgot Password (normal) ---- */
//                            composable(Screen.EditPass.route) {
//                                EditPassPage(onBackToLogin = { navController.popBackStack() })
//                            }
//
//                            /* ---- Private pages (dengan TopAppBar + Drawer) ---- */
//                            composable(Screen.Home.route) {
//                                HomePage(
//                                    onMenuClick = { scope.launch { drawerState.open() } }
//                                )
//                            }
//                            composable(Screen.Live.route) { LiveParkingPage() }
//                            composable(Screen.History.route) { HistoryPage() }
//                            composable(Screen.Info.route) { InformationPage() }
//                            composable(Screen.Logout.route) {
//                                LogoutPage(
//                                    onCancel = { navController.popBackStack() },
//                                    onLoggedOut = {
//                                        scope.launch {
//                                            repo.logout() // hapus row session → Flow akan jadi null → startDest jadi Landing/Login
//                                            navController.navigate(Screen.Login.route) {
//                                                popUpTo(Screen.Landing.route) { inclusive = true }
//                                                launchSingleTop = true
//                                            }
//                                        }
//                                    }
//                                )
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.room.Room
import com.example.smartparking.data.network.TokenProvider
import com.example.smartparking.ui.components.DrawerContent
import com.example.smartparking.ui.editpasspage.EditPassPage
import com.example.smartparking.ui.historypage.HistoryPage
import com.example.smartparking.ui.homepage.HomePage
import com.example.smartparking.ui.informationpage.InformationPage
import com.example.smartparking.ui.landingpage.LandingPageScreen
import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
import com.example.smartparking.data.repository.db.AppDatabase
import com.example.smartparking.ui.loginpage.LoginPage
import com.example.smartparking.ui.loginpage.LoginVMFactory
import com.example.smartparking.ui.loginpage.LoginViewModel
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
    userName: String?,
    userEmail: String?,
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
                    scope.launch {
                        drawerState.close()
                        delay(60)
                        onNavigateRoute(route)
                    }
                },
                userName,
                userEmail
            )
        }
    ) {
        Box(Modifier.fillMaxSize()) {
            content { scope.launch { drawerState.open() } }

            // Hamburger melayang global
            Surface(
                tonalElevation = 3.dp,
                shadowElevation = 6.dp,
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.65f),
                shape = MaterialTheme.shapes.extraLarge,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(start = 12.dp, top = 50.dp)
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
                val appCtx = applicationContext

                val db = remember {
                    Room.databaseBuilder(appCtx, AppDatabase::class.java, "smartparking.db").build()
                }
                val repo = remember { com.example.smartparking.data.repository.UserRepository(db.sessionDao()) }

                val sessionFlow = remember { db.sessionDao().observeSession() }
                val session by sessionFlow.collectAsStateWithLifecycle(initialValue = null)

                LaunchedEffect(Unit) {
                    TokenProvider.init(db.sessionDao())
                }


                val startDest = if (session == null) Screen.Landing.route else Screen.Home.route


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
                            startDestination = startDest,
                            modifier = Modifier.fillMaxSize()
                        ) {
                            // ---------- Public ----------
                            composable(Screen.Landing.route) {
                                LandingPageScreen(
                                    brandName = "SPARK",
                                    subTitle = "Smart Parking FT UGM",
                                    brandColor = Color(0xFF0A2342),
                                    modifier = Modifier.fillMaxSize(),
                                    onNavigateNext = { navController.navigate(Screen.Login.route) }
                                )
                            }
                            composable(Screen.Login.route) {
                                val vm: LoginViewModel =
                                    viewModel(
                                        factory = LoginVMFactory(repo)
                                    )

                                LoginPage(
                                    vm = vm,
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

                            // ---------- Private (pakai drawer global) ----------
                            composable(Screen.Home.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Home.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    userName = session?.name,          // ⬅️ dari Room
                                    userEmail = session?.email         // ⬅️ dari Room
                                ) { _ ->
                                    HomePage()
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
                                    },
                                    userName = session?.name,
                                    userEmail = session?.email
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
                                    },
                                    userName = session?.name,
                                    userEmail = session?.email
                                ) { _ -> HistoryPage() }
                            }
                            composable(Screen.Info.route) {
                                WithDrawer(
                                    selectedRoute = Screen.Info.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    userName = session?.name,
                                    userEmail = session?.email
                                ) { _ -> InformationPage() }
                            }
                            composable(Screen.Logout.route) {
                                val scope = rememberCoroutineScope()
                                WithDrawer(
                                    selectedRoute = Screen.Logout.route,
                                    onNavigateRoute = { route ->
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(Screen.Home.route) { inclusive = false }
                                        }
                                    },
                                    userName = session?.name,
                                    userEmail = session?.email
                                ) { _ ->
                                    LogoutPage(
                                        onCancel = { navController.popBackStack() },
                                        onLoggedOut = {
                                            scope.launch {
                                                repo.logout()
                                                navController.navigate(Screen.Login.route) {
                                                    popUpTo(Screen.Landing.route) { inclusive = true }
                                                    launchSingleTop = true
                                                }
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

//package com.example.smartparking
//
//import android.os.Bundle
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.enableEdgeToEdge
//import androidx.activity.viewModels
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.padding
//import androidx.compose.material3.Scaffold
//import androidx.compose.runtime.getValue
//import androidx.compose.runtime.collectAsState
//import androidx.compose.ui.Modifier
//import androidx.navigation.NavController
//import androidx.navigation.compose.rememberNavController
//import com.example.smartparking.ui.beacontest.BeaconScreen
//import com.example.smartparking.ui.home.HomeScreen
////import com.example.smartparking.ui.home.HomeScreenPreview
//import com.example.smartparking.ui.home.HomeViewModel
//import com.example.smartparking.ui.loginpage.LoginPage
//import com.example.smartparking.ui.loginpage.LoginViewModel
//import com.example.smartparking.ui.theme.SmartParkingTheme
//
//class MainActivity : ComponentActivity() {
//
//    // Attach ViewModel to the activity
//    private val homeViewModel: HomeViewModel by viewModels()
//    private val loginViewModel: LoginViewModel by viewModels()
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        enableEdgeToEdge()
//        setContent {
//            SmartParkingTheme {
//                val state by homeViewModel.uiState.collectAsState()
//                val navController = rememberNavController()
//                BeaconScreen()
//
////                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
//////                    HomeScreen(
//////                        viewModel = homeViewModel,
//////                        modifier = Modifier.padding(innerPadding)
//////                    )
//////                    LoginScreen(navController, loginViewModel)
////                }
//
//            }
//        }
//    }
//}
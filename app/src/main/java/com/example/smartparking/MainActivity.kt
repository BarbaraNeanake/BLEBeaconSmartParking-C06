package com.example.smartparking

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.smartparking.data.network.TokenProvider
import com.example.smartparking.data.remote.RetrofitProvider
import com.example.smartparking.data.repository.UserRepository
import com.example.smartparking.data.repository.db.AppDatabase
import com.example.smartparking.ui.beacontest.BeaconViewModel
import com.example.smartparking.ui.components.DrawerContent
import com.example.smartparking.ui.editpasspage.EditPassPage
import com.example.smartparking.ui.editpasspage.EditPassVMFactory
import com.example.smartparking.ui.editpasspage.EditPassViewModel
import com.example.smartparking.ui.historypage.HistoryPage
import com.example.smartparking.ui.homepage.HomePage
import com.example.smartparking.ui.informationpage.InformationPage
import com.example.smartparking.ui.landingpage.LandingPageScreen
import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
import com.example.smartparking.ui.liveparkingpage.LiveParkingVMFactory
import com.example.smartparking.ui.liveparkingpage.LiveParkingViewModel
import com.example.smartparking.ui.loginpage.LoginPage
import com.example.smartparking.ui.loginpage.LoginVMFactory
import com.example.smartparking.ui.loginpage.LoginViewModel
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

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            SmartParkingTheme {
                // ---------------- Core instances ----------------
                val navController = rememberNavController()
                val backStackEntry by navController.currentBackStackEntryAsState()
                val currentRoute = backStackEntry?.destination?.route

                val appCtx = applicationContext
                val db = remember { AppDatabase.getInstance(appCtx) }

                // Session untuk startDestination & header drawer
                val sessionFlow = remember { db.sessionDao().observeSession() }
                val session by sessionFlow.collectAsStateWithLifecycle(initialValue = null)

                // Init TokenProvider (untuk AuthInterceptor)
                LaunchedEffect(Unit) { TokenProvider.init(db.sessionDao()) }

                // Drawer hanya aktif di private routes
                val privateRoutes = remember {
                    setOf(Screen.Home.route, Screen.Live.route, Screen.History.route, Screen.Info.route, Screen.Logout.route)
                }
                val isPrivate = currentRoute in privateRoutes
                val drawerState = rememberDrawerState(DrawerValue.Closed)
                val scope = rememberCoroutineScope()

                // ---------------- BLE: auto scan on app start ----------------
                val beaconVm: BeaconViewModel = viewModel()

                // permissions (API-level aware)
                val blePermissions: Array<String> = remember {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                        arrayOf(
                            Manifest.permission.BLUETOOTH_SCAN,
                            Manifest.permission.BLUETOOTH_CONNECT
                        )
                    } else {
                        arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
                    }
                }

                var permissionsGranted by remember { mutableStateOf(false) }
                var btEnabled by remember { mutableStateOf(BluetoothAdapter.getDefaultAdapter()?.isEnabled == true) }
                var scanStarted by remember { mutableStateOf(false) }

                val permissionLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestMultiplePermissions()
                ) { grants ->
                    permissionsGranted = grants.values.all { it }
                }

                val enableBtLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.StartActivityForResult()
                ) {
                    btEnabled = BluetoothAdapter.getDefaultAdapter()?.isEnabled == true
                }

                fun hasBlePermissions(): Boolean {
                    return blePermissions.all { p ->
                        ContextCompat.checkSelfPermission(this, p) == PackageManager.PERMISSION_GRANTED
                    }
                }

                // Request permission saat start
                LaunchedEffect(Unit) {
                    permissionsGranted = hasBlePermissions()
                    if (!permissionsGranted) permissionLauncher.launch(blePermissions)
                }

                // Minta enable BT bila perlu
                LaunchedEffect(permissionsGranted) {
                    if (permissionsGranted && !btEnabled) {
                        enableBtLauncher.launch(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE))
                    }
                }

                // Start scan sekali saja saat sudah OK
                LaunchedEffect(permissionsGranted, btEnabled) {
                    if (permissionsGranted && btEnabled && !scanStarted) {
                        scanStarted = true
                        beaconVm.startScan()
                    }
                }

                // Stop scan saat composition dibuang
                DisposableEffect(Unit) {
                    onDispose { beaconVm.stopScan() }
                }

                // ---------------- Bridge BLE → Parking update ----------------
                val liveVm: LiveParkingViewModel = viewModel(factory = LiveParkingVMFactory(db.sessionDao()))
                val userId: Int? = session?.userId
                val userRole: String? = session?.role

                val api = remember { com.example.smartparking.data.repository.UserRepository(db.sessionDao()) }
                val vm: EditPassViewModel = viewModel(factory = EditPassVMFactory(api))

                LaunchedEffect(Unit) {
                    // Ketika slot terdeteksi, update DB lalu reload warna
                    beaconVm.detectedSlot.collect { slotId ->
                        liveVm.applyBeaconDetection(slotId)
                    }

                }

                // ---------------- UI scaffold + Navigation ----------------
                ModalNavigationDrawer(
                    drawerState = drawerState,
                    gesturesEnabled = isPrivate,
                    drawerContent = {
                        if (isPrivate) {
                            com.example.smartparking.ui.components.DrawerContent(
                                selectedRoute = currentRoute,
                                onItemClick = { route ->
                                    scope.launch {
                                        // tutup drawer cepat agar aman saat navigate
                                        drawerState.close()
                                        navController.navigate(route) {
                                            launchSingleTop = true
                                            popUpTo(navController.graph.findStartDestination().id) { inclusive = false }
                                        }
                                    }
                                },
                                userName = session?.name ?: "-",
                                userEmail = session?.email ?: "-"
                            )
                        }
                    }
                ) {
                    Scaffold(
                        containerColor = Color.Transparent,
                        contentWindowInsets = WindowInsets(0),
//                        topBar = {
//                            if (isPrivate) {
//                                TopAppBar(
//                                    title = {
//                                        Text(
//                                            when (currentRoute) {
//                                                Screen.Home.route -> "Home"
//                                                Screen.Live.route -> "Live Parking"
//                                                Screen.History.route -> "History"
//                                                Screen.Info.route -> "Information"
//                                                Screen.Logout.route -> "Logout"
//                                                else -> ""
//                                            }
//                                        )
//                                    },
//                                    navigationIcon = {
//                                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
//                                            Icon(Icons.Filled.Menu, contentDescription = "Menu")
//                                        }
//                                    }
//                                )
//                            }
//                        }
                    ) { innerPadding ->
                        Box(
                            Modifier
                                .fillMaxSize()
                                .padding(innerPadding)
                        ) {
                            NavHost(
                                navController = navController,
                                startDestination = if (session == null) Screen.Landing.route else Screen.Home.route,
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
                                    val userRepo = remember { UserRepository(db.sessionDao()) }
                                    val loginVm: LoginViewModel = viewModel(factory = LoginVMFactory(userRepo))
                                    LoginPage(
                                        vm = loginVm,
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
                                            navController.navigate(Screen.Login.route) {
                                                launchSingleTop = true
                                            }
                                        },
                                        onBackToLogin = { navController.popBackStack() }
                                    )
                                }
                                composable(Screen.EditPass.route) {
                                    EditPassPage(
                                        vm = vm,
                                        onBackToLogin = {
                                            // balik ke login setelah success
                                            navController.popBackStack(Screen.Login.route, inclusive = false)
                                        }
                                    )
                                }

                                // ---------- Private ----------
                                composable(Screen.Home.route) {
                                    HomePage()
                                }
                                composable(Screen.Live.route) {
                                    // pakai liveVm yang sama (satu instance di activity scope)
                                    LiveParkingPage(vm = liveVm)
                                }
                                composable(Screen.History.route) {
                                    HistoryPage()
                                }
                                composable(Screen.Info.route) {
                                    InformationPage()
                                }
                                composable(Screen.Logout.route) {
                                    val userRepo = remember { UserRepository(db.sessionDao()) }
                                    LogoutPage(
                                        onCancel = { navController.popBackStack() },
                                        onLoggedOut = {
                                            scope.launch {
                                                userRepo.logout()
                                                drawerState.close()
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

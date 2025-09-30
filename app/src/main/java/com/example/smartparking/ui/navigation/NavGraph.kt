package com.example.smartparking.ui.navigation

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import com.google.accompanist.navigation.animation.AnimatedNavHost
import com.google.accompanist.navigation.animation.composable
import com.example.smartparking.ui.landingpage.LandingPage
import com.example.smartparking.ui.loginpage.LoginPage
import com.example.smartparking.ui.signuppage.SignUpPage
import com.example.smartparking.ui.homepage.HomePage
import com.example.smartparking.ui.liveparkingpage.LiveParkingPage
import com.example.smartparking.ui.informationpage.InformationPage
import com.example.smartparking.ui.historypage.HistoryPage
import com.example.smartparking.ui.logoutpage.LogoutPage

private object Routes {
    const val LANDING = "landing"
    const val LOGIN = "login"
    const val SIGN_UP = "signup"
    const val HOME = "home"
    const val LIVE_PARKING = "live_parking"
    const val INFO = "information"
    const val HISTORY = "history"
    const val LOGOUT = "logout"
}

@OptIn(ExperimentalAnimationApi::class)
@Composable
fun NavGraph(
    navController: NavHostController = rememberNavController()
) {
    AnimatedNavHost(
        navController = navController,
        startDestination = Routes.LANDING
    ) {
        // Landing
        composable(
            route = Routes.LANDING,
            enterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            exitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            popEnterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right) },
            popExitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right) }
        ) {
            LandingPage(
                appName = "Smart Parking FT UGM",
                onNavigateNext = { navController.navigate(Routes.LOGIN) }
            )
        }

        // Login
        composable(
            route = Routes.LOGIN,
            enterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            exitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            popEnterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right) },
            popExitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right) }
        ) {
            LoginPage(
                onLoginSuccess = { navController.navigate(Routes.HOME) },
                onSignUpClick = { navController.navigate(Routes.SIGN_UP) }
            )
        }

        // Sign Up
        composable(
            route = Routes.SIGN_UP,
            enterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            exitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left) },
            popEnterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right) },
            popExitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right) }
        ) {
            SignUpPage(
                onRegistered = { navController.navigate(Routes.HOME) },
                onBackToLogin = { navController.popBackStack() }
            )
        }

        // Home
        composable(route = Routes.HOME) {
            HomePage()
        }

        // Live Parking
        composable(route = Routes.LIVE_PARKING) {
            LiveParkingPage()
        }

        // Information
        composable(route = Routes.INFO) {
            InformationPage()
        }

        // History
        composable(route = Routes.HISTORY) {
            HistoryPage()
        }

        // Logout (popup konfirmasi)
        composable(route = Routes.LOGOUT) {
            LogoutPage(
                onCancel = { navController.popBackStack() },
                onLoggedOut = {
                    navController.navigate(Routes.LOGIN) {
                        // bersihkan back stack supaya tidak bisa back ke halaman private
                        popUpTo(Routes.LANDING) { inclusive = true }
                        launchSingleTop = true
                    }
                }
            )
        }
    }
}

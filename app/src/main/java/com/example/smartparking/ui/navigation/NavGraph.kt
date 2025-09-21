package com.example.smartparking.ui.navigation

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import com.google.accompanist.navigation.animation.AnimatedNavHost
import com.google.accompanist.navigation.animation.composable
import com.example.smartparking.ui.landingpage.LandingPage
import com.example.smartparking.ui.loginpage.LoginPage

@OptIn(ExperimentalAnimationApi::class)
@Composable
fun NavGraph(
    navController: NavHostController = rememberNavController()
) {
    AnimatedNavHost(
        navController = navController,
        startDestination = Routes.LANDING
    ) {
        composable(
            route = Routes.LANDING,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left)
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left)
            },
            popEnterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right)
            },
            popExitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right)
            }
        ) {
            LandingPage(
                appName = "Smart Parking FT UGM",
                onNavigateNext = { navController.navigate(Routes.LOGIN) }
            )
        }

        composable(
            route = Routes.LOGIN,
            enterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left)
            },
            exitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left)
            },
            popEnterTransition = {
                slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right)
            },
            popExitTransition = {
                slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right)
            }
        ) {
            LoginPage(
                onLoginSuccess = {
                    navController.navigate(Routes.HOME)
                }
            )
        }

        // Placeholder HOME biar compile (nanti ganti ke HomeScreen kamu)
        composable(route = Routes.HOME) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Home")
            }
        }
    }
}

private object Routes {
    const val LANDING = "landing"
    const val LOGIN = "login"
    const val HOME = "home"
}

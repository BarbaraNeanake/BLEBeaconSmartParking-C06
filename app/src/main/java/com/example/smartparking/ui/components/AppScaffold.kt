package com.example.smartparking.ui.components

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScaffold(
    selectedRoute: String?,
    onNavigate: (String) -> Unit,
    topBarTitle: String = "",
    userName: String? = null,
    userEmail: String? = null,
    content: @Composable () -> Unit
) {
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = true,
        drawerContent = {
            DrawerContent(
                selectedRoute = selectedRoute,
                onItemClick = { route ->
                    scope.launch {
                        // ⬇️ TUTUP TANPA ANIMASI agar tidak kehilangan density saat navigate
                        drawerState.snapTo(DrawerValue.Closed)
                        onNavigate(route)
                    }
                },
                userName = userName ?: "-",
                userEmail = userEmail ?: "-"
            )
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(topBarTitle) },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Menu")
                        }
                    }
                )
            }
        ) { innerPadding ->
            Surface(modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
            ) {
                content()
            }
        }
    }
}

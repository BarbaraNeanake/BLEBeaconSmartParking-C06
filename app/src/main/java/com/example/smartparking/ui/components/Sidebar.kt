package com.example.smartparking.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Map
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R

data class DrawerItem(
    val label: String,
    val icon: ImageVector,
    val route: String
)

@Composable
fun DrawerContent(
    selectedRoute: String?,
    onItemClick: (route: String) -> Unit,
    userName: String = "Barbara Neanake",
    userEmail: String = "barbaraneanake@ugm.ac.id"
) {
    val items = listOf(
        DrawerItem("Home", Icons.Filled.Home, "home"),
        DrawerItem("Live Parking Map", Icons.Filled.Map, "live_parking"),
        DrawerItem("History", Icons.Filled.History, "history"),
        DrawerItem("Information", Icons.Filled.Info, "information"),
        DrawerItem("Logout", Icons.Filled.ExitToApp, "logout"),
    )

    ModalDrawerSheet(
        drawerContainerColor = MaterialTheme.colorScheme.surface,
        drawerContentColor = MaterialTheme.colorScheme.onSurface
    ) {
        // Header
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 24.dp, bottom = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.size(56.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text("Smart Parking", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Divider()
        }

        // Items
        items.forEach { item ->
            val selected = selectedRoute == item.route
            NavigationDrawerItem(
                label = { Text(item.label, fontSize = 16.sp) },
                selected = selected,
                onClick = { onItemClick(item.route) },
                icon = { Icon(item.icon, contentDescription = item.label) },
                modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
            )
        }

        Spacer(Modifier.weight(1f))

        // Footer / user panel
        Divider()
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // pakai logo bulat sebagai placeholder avatar
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "Avatar",
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
            )
            Spacer(Modifier.width(10.dp))
            Column {
                Text(userName, style = MaterialTheme.typography.bodyMedium)
                Text(userEmail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

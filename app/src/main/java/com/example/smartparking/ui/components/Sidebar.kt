package com.example.smartparking.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Map
import androidx.compose.material.icons.outlined.PowerSettingsNew
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.smartparking.R

data class DrawerItem(
    val label: String,
    val route: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)

@Composable
fun DrawerContent(
    selectedRoute: String?,
    onItemClick: (route: String) -> Unit,
    userName: String = "Barbara Neanake",
    userEmail: String = "barbaraneanake@ugm.ac.id",
    drawerWidthFraction: Float = 0.78f
) {
    val items = listOf(
        DrawerItem("Home", "home", Icons.Outlined.Home),
        DrawerItem("Live Parking Map", "live_parking", Icons.Outlined.Map),
        DrawerItem("History", "history", Icons.Outlined.History),
        DrawerItem("Information", "information", Icons.Outlined.Info),
        DrawerItem("Logout", "logout", Icons.Outlined.PowerSettingsNew)
    )

    ModalDrawerSheet(
        modifier = Modifier.fillMaxWidth(drawerWidthFraction)
    ) {
        // ===== Header =====
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 18.dp, bottom = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.ugm_logo),
                contentDescription = "UGM Logo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.size(54.dp)
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = "SPARK",
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
            )
            Spacer(Modifier.height(6.dp))
            HorizontalDivider()
        }

        // ===== Items =====
        items.forEach { item ->
            val selected = selectedRoute == item.route
            NavigationDrawerItem(
                label = { Text(item.label, fontSize = 16.sp) },
                icon = { Icon(item.icon, contentDescription = item.label) },
                selected = selected,
                onClick = { onItemClick(item.route) },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                colors = NavigationDrawerItemDefaults.colors(
                    selectedContainerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                    selectedTextColor = MaterialTheme.colorScheme.primary,
                    selectedIconColor = MaterialTheme.colorScheme.primary
                )
            )
        }

        Spacer(modifier = Modifier.weight(1f)) // ini aman karena berada di Column scope

        // ===== Footer (User Panel) =====
        HorizontalDivider()
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                painter = painterResource(id = R.drawable.user), // pastikan user.png ada di res/drawable
                contentDescription = "User Avatar",
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .size(38.dp)
                    .clip(CircleShape)
            )
            Spacer(Modifier.width(10.dp))
            Column {
                Text(userName, style = MaterialTheme.typography.bodyMedium)
                Text(
                    userEmail,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

//package com.example.smartparking.ui.beacontest
//
//import android.Manifest
//import android.widget.Toast
//import androidx.activity.compose.rememberLauncherForActivityResult
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.lazy.LazyColumn
//import androidx.compose.foundation.lazy.items
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.platform.LocalContext
//import androidx.compose.ui.unit.dp
//import androidx.lifecycle.viewmodel.compose.viewModel
//
//@Composable
//fun BeaconScreen(viewModel: BeaconViewModel = viewModel()) {
//    val context = LocalContext.current
//    val beacons by viewModel.beacons.collectAsState()
//    val detectedSlot by viewModel.detectedSlot.collectAsState()
//    val assignResult by viewModel.assignResult.collectAsState()
//
//    val permissionLauncher = rememberPermissionLauncher {
//        viewModel.startScan()
//    }
//
//    // Tampilkan Toast saat ada hasil assign baru
//    LaunchedEffect(assignResult) {
//        if (assignResult.isNotEmpty()) {
//            Toast.makeText(context, assignResult, Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp)
//    ) {
//        ControlButtons(
//            onStart = { permissionLauncher.launchPermissions() },
//            onStop = { viewModel.stopScan() }
//        )
//
//        Spacer(modifier = Modifier.height(16.dp))
//
//        Text(
//            text = "Posisi Anda: $detectedSlot",
//            style = MaterialTheme.typography.titleLarge
//        )
//
//        Spacer(modifier = Modifier.height(8.dp))
//
//        AssignButton { viewModel.assignSlot(userId = 2, userRole = "Dosen") }
//
//        Spacer(modifier = Modifier.height(16.dp))
//
//        BeaconList(beacons)
//    }
//}
//
//@Composable
//private fun ControlButtons(onStart: () -> Unit, onStop: () -> Unit) {
//    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
//        Button(onClick = onStart) { Text("Start Scanning") }
//        Button(onClick = onStop) { Text("Stop") }
//    }
//}
//
//@Composable
//private fun AssignButton(onAssign: () -> Unit) {
//    Button(onClick = onAssign) {
//        Text("Assign Slot")
//    }
//}
//
//@Composable
//private fun BeaconList(beacons: List<BeaconData>) {
//    LazyColumn(
//        modifier = Modifier.fillMaxSize(),
//        verticalArrangement = Arrangement.spacedBy(8.dp)
//    ) {
//        items(beacons) { beacon ->
//            BeaconItem(beacon)
//        }
//    }
//}
//
//@Composable
//private fun BeaconItem(beacon: BeaconData) {
//    Card(modifier = Modifier.fillMaxWidth()) {
//        Column(modifier = Modifier.padding(12.dp)) {
//            Text("Name: ${beacon.name}")
//            Text("Address: ${beacon.address}")
//            Text("RSSI Samples: ${beacon.rssiSamples}")
//        }
//    }
//}
//
//// Helper untuk permission
//@Composable
//private fun rememberPermissionLauncher(onAllGranted: () -> Unit): PermissionLauncher {
//    val context = LocalContext.current
//    val launcher = rememberLauncherForActivityResult(
//        contract = ActivityResultContracts.RequestMultiplePermissions()
//    ) { permissions ->
//        if (permissions.all { it.value }) {
//            onAllGranted()
//        } else {
//            Toast.makeText(context, "Permission required to scan", Toast.LENGTH_SHORT).show()
//        }
//    }
//    return PermissionLauncher { launcher.launch(arrayOf(
//        Manifest.permission.BLUETOOTH_SCAN,
//        Manifest.permission.BLUETOOTH_CONNECT,
//        Manifest.permission.ACCESS_FINE_LOCATION
//    )) }
//}
//
//class PermissionLauncher(private val launchAction: () -> Unit) {
//    fun launchPermissions() = launchAction()
//}

package com.example.smartparking.ui.beacontest

import android.Manifest
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun BeaconScreen(viewModel: BeaconViewModel = viewModel()) {
    val context = LocalContext.current
    val beacons by viewModel.beacons.collectAsState()
    val detectedSlot by viewModel.detectedSlot.collectAsState()

    // Launcher untuk request permission
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            viewModel.startScan()
        } else {
            Toast.makeText(context, "Permission required to scan", Toast.LENGTH_SHORT).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                launcher.launch(
                    arrayOf(
                        Manifest.permission.BLUETOOTH_SCAN,
                        Manifest.permission.BLUETOOTH_CONNECT,
                        Manifest.permission.ACCESS_FINE_LOCATION
                    )
                )
            }) {
                Text("Start Scanning")
            }

            Button(onClick = { viewModel.stopScan() }) {
                Text("Stop")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Posisi Anda: $detectedSlot",
            style = MaterialTheme.typography.titleLarge
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(beacons) { beacon ->
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text("Name: ${beacon.name}")
                        Text("Address: ${beacon.address}")
                        Text("RSSI Samples: ${beacon.rssiSamples}")
                    }
                }
            }
        }
    }
}
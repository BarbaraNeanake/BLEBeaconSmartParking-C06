System Description: IoT Parking Management Model

1. Project Overview

This project is a Real-Time Parking Management System designed to detect the presence of miniature cars in specific parking slots and broadcast their availability status to the cloud. It utilizes an STM32 Blackpill microcontroller as the central processing unit and an ESP-01 module for Wi-Fi communication.

2. Working Principle

The system operates on a magnetic detection mechanism:

The Setup: Each parking slot is equipped with a Magnetic Reed Switch (Normally Open) buried in the ground.

The Trigger: Each miniature car has a small magnet attached to its chassis.

Detection:

Slot Empty: The switch remains open. The STM32 reads a HIGH signal (3.3V) via its internal pull-up resistors.

Slot Occupied: When a car parks, the magnet closes the reed switch, connecting the circuit to Ground. The STM32 reads a LOW signal (0V).

3. Data Transmission

Upon detecting a status change (e.g., a car arriving or leaving), the STM32 sends an AT command to the ESP-01. The ESP-01 then publishes an MQTT message to the HiveMQ Public Broker.

Protocol: MQTT over TCP/IP.

Topic: SPARK_C06/isOccupied/{Slot_ID}

Payload: "True" (Occupied) or "False" (Available).

4. Hardware Specifications

Controller: STM32F401/F411 "Blackpill" (Logic & Sensor Reading).

Network Module: ESP-01 (ESP8266) communicating via UART (Serial).

Sensors: 5x Passive Magnetic Reed Switches.

Power Logic: Active Low (Ground triggering) for electrical safety and noise immunity.
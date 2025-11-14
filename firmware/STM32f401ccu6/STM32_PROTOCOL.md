# STM32F401CCU6 Firmware Protocol Documentation

## Overview
This firmware implements a smart parking IoT system using STM32F401CCU6 (Black Pill) as an intelligent debouncing controller. The system receives occupancy sensor data via MQTT (ESP-01), implements a 4-state stabilization machine to handle parking alignment maneuvers, and triggers ESP32-CAM to capture and upload parking events.

**Hardware:** STM32F401CCU6 (64KB RAM, 256KB Flash, 84MHz)  
**Firmware Version:** 2.0 - Smart Parking State Machine  
**Date:** November 12, 2025

---

## Key Architectural Changes (v2.0)

### What Changed from v1.0?

**v1.0 Architecture (DEPRECATED):**
```
Sensor → MQTT → STM32 → SPI → ESP32-CAM
                  ↓
            Image buffer (76KB)
                  ↓
            ESP-01 → HTTP POST
```
**Problems:**
- ❌ ESP-01 doesn't support HTTPS (backend requires HTTPS)
- ❌ STM32 buffers entire image (76KB RAM usage)
- ❌ UART bottleneck @ 115200 baud (~9.6 seconds)
- ❌ Simple trigger → multiple captures per parking event
- ❌ No debouncing for parking alignment maneuvers

**v2.0 Architecture (CURRENT):**
```
Sensor → MQTT → STM32 (State Machine) → 100ms pulse → ESP32-CAM
                  ↓                                        ↓
            Debouncing logic                    Capture + WiFi + HTTPS
            (2s + 5s adaptive)                             ↓
                                                        Backend
```
**Benefits:**
- ✅ ESP32-CAM handles HTTPS natively (backend compatible)
- ✅ STM32 only stores state (~12 bytes, not 76KB)
- ✅ No UART bottleneck (ESP32 native WiFi)
- ✅ Smart debouncing → ONE image per parking event
- ✅ 4-state machine handles real parking behavior
- ✅ Multi-slot monitoring with wildcard subscription
- ✅ Adaptive cooldown exits early when car leaves
- ✅ Fast response (2s stabilization, 5s cooldown)
- ✅ 43% RAM reduction (776 bytes vs 1,372 bytes)
- ✅ 22% Flash reduction (10,012 bytes vs 12,756 bytes)

### Role Separation

| Component | v1.0 Role | v2.0 Role |
|-----------|-----------|-----------|
| **STM32** | Controller + Image Buffer + HTTP Client | **State Machine Controller Only** |
| **ESP-01** | WiFi + MQTT + HTTP Transport | **MQTT Gateway Only** |
| **ESP32-CAM** | Camera Sensor Only | **Camera + WiFi + JPEG + HTTPS + Processing** |

### Why State Machine?

**Real-World Parking Scenario:**
1. Driver approaches slot → sensor detects (1st trigger)
2. Driver reverses to align → sensor clears momentarily
3. Driver moves forward → sensor detects again (2nd trigger)
4. Driver adjusts angle → sensor fluctuates (3rd, 4th triggers)
5. Car finally parked → sensor stable for 2 seconds

**Without Debouncing:** 4+ images captured  
**With State Machine:** 1 image captured (after 2s stability)

**Multi-Slot Consideration:**
- Camera monitors multiple slots (A01, A02, A03) with wide-angle lens
- If multiple cars arrive within 2s window, last car to stabilize triggers capture
- Single image captures ALL cars in view (backend processes to detect individual slots)
- Adaptive cooldown: exits early when car leaves (allows fast queue processing)

---

## System Architecture

```
┌──────────────────┐
│  On-Slot Sensors │ (Multiple slots: A01, A02, A03, ...)
│  (Occupancy)     │
└────────┬─────────┘
         │ MQTT: SPARK_C06/isOccupied/{slotID}
         │ Payload: "1" (occupied) / "0" (empty)
         ▼
┌─────────────┐        ┌─────────────────────┐        ┌─────────────┐
│   ESP-01    │◄──────►│     STM32F401       │───────►│ ESP32-CAM   │
│ (WiFi/MQTT) │  UART  │ (State Machine)     │ Trigger│  (Camera)   │
│             │ 115200 │                     │ PB12   │             │
└─────────────┘        │  States:            │ 100ms  └─────────────┘
      ▲                │  • IDLE             │ pulse        │
      │                │  • STABILIZING (2s) │              │
┌─────────────┐        │  • CONFIRMED        │              │
│ MQTT Broker │        │  • COOLDOWN (5s)    │              │
│ HiveMQ.com  │        └─────────┬───────────┘              │
└─────────────┘                  │                          │
                                 │ PC13 LED                 │ WiFi + HTTPS
                                 ▼                          ▼
                          ┌─────────────┐          ┌──────────────┐
                          │  Status LED │          │   Backend    │
                          │  Indicator  │          │ Hugging Face │
                          └─────────────┘          └──────────────┘
```

**Smart Parking Flow:**
1. On-slot sensor publishes "1" → STM32 enters STABILIZING
2. Additional "1" messages → Reset 2s timer (car still aligning)
3. "0" message during STABILIZING → Return to IDLE (car left)
4. 2s stable → CONFIRMED → Trigger ESP32-CAM (100ms pulse)
5. ESP32-CAM captures VGA image → JPEG compress → HTTPS POST
6. STM32 enters COOLDOWN (5s minimum) → Ignore all "1" messages
7. If "0" received during COOLDOWN → Exit early to IDLE (fast queue processing)
8. After 5s (or early exit) → Return to IDLE (ready for next parking event)

---

## Pin Configuration

### ESP-01 (WiFi Module) - UART Interface
| STM32 Pin | Function | ESP-01 Pin | Description |
|-----------|----------|------------|-------------|
| PB6 | USART1_TX | RX | Data to ESP-01 |
| PB7 | USART1_RX | TX | Data from ESP-01 |
| GND | Ground | GND | Common ground |
| 3.3V | Power | VCC/CH_PD | Power supply |

**Baud Rate:** 115200  
**Protocol:** AT Commands + Raw TCP/IP

### ESP32-CAM (Camera Module) - Trigger Interface
| STM32 Pin | Function | ESP32-CAM Pin | Description |
|-----------|----------|---------------|-------------|
| PB12 | GPIO Output | GPIO15 | **Trigger Signal (100ms HIGH pulse)** |
| PB13 | SPI1_SCK | GPIO12 | SPI Clock (PCB trace present, unused) |
| PB14 | SPI1_MISO | GPIO14 | Data from Camera (PCB trace present, unused) |
| PB15 | SPI1_MOSI | GPIO13 | Data to Camera (PCB trace present, unused) |
| GND | Ground | GND | Common ground |
| 5V | Power | 5V | Power from booster (via 470µF + 100µF caps) |

**Current Implementation:**
- **Trigger Protocol:** 100ms HIGH pulse on PB12 → ESP32-CAM GPIO15
- **SPI Traces:** Present on PCB for future bidirectional communication
- **Data Flow:** ESP32-CAM handles capture → WiFi → HTTPS upload independently
- **No SPI Transfer:** Image data never sent to STM32 (ESP32 has 520KB SRAM)

### Status Indicator
| STM32 Pin | Function | Description |
|-----------|----------|-------------|
| PC13 | LED Output | Onboard LED (Active HIGH) |

---

## Communication Protocols

### 1. MQTT Protocol (via ESP-01)

#### Connection Flow
```
1. WiFi Connection
   AT+RST
   AT+CWMODE=1
   AT+CWJAP="SSID","PASSWORD"

2. TCP Connection to Broker
   AT+CIPSTART="TCP","broker.hivemq.com",1883

3. MQTT CONNECT Packet
   [0x10][remaining_len][protocol_name][level][flags][keep_alive][client_id]

4. MQTT SUBSCRIBE Packet
   [0x82][remaining_len][packet_id][topic_length][topic][qos]
```

#### MQTT Packet Structure

**CONNECT Packet:**
```c
0x10                    // Packet type: CONNECT
<remaining_length>      // Variable length
0x00 0x04               // Protocol name length
'M' 'Q' 'T' 'T'        // Protocol name
0x04                    // Protocol level (MQTT 3.1.1)
0x02                    // Connect flags (Clean session)
0x00 0x3C               // Keep alive (60 seconds)
<client_id_length>      // Client ID length (2 bytes)
<client_id>             // Client ID string
```

**SUBSCRIBE Packet:**
```c
0x82                    // Packet type: SUBSCRIBE with QoS 1
<remaining_length>      // Variable length
0x00 0x01               // Packet ID
0x00 <topic_len>        // Topic length (2 bytes)
<topic>                 // Topic string
0x00                    // Requested QoS level
```

**PUBLISH Packet (Received):**
```c
0x30                    // Packet type: PUBLISH (QoS 0)
<remaining_length>      // Variable length
<topic_length>          // Topic length (2 bytes)
<topic>                 // Topic string
<payload>               // Message payload
```

**PINGREQ Packet (Keep-Alive):**
```c
0xC0 0x00              // PINGREQ (sent every 30 seconds)
```

#### MQTT Topics (Smart Parking System)
- **Subscribe:** `SPARK_C06/isOccupied/*` (wildcard - monitors ALL parking slots)
  - Example messages: `SPARK_C06/isOccupied/A01`, `SPARK_C06/isOccupied/B12`
- **Publish:** `SPARK_C06/camera/status` (camera system status)
- **Client ID:** `SPARK_C06`
- **Broker:** `broker.hivemq.com:1883`

#### Message Payloads
- **"1"** - Slot occupied (car detected by on-slot sensor)
- **"0"** - Slot empty (no car present)

#### Trigger Mechanism (State Machine)
The system implements intelligent debouncing to handle real-world parking scenarios:

1. **IDLE State:** Waiting for first "1" message
2. **STABILIZING State (2s):** 
   - Car detected, waiting for driver to finish aligning
   - Additional "1" messages reset the 2-second timer
   - "0" message returns to IDLE (car left before parking)
3. **CONFIRMED State:** 
   - 2 seconds elapsed with stable occupancy
   - Trigger ESP32-CAM with 100ms HIGH pulse on PB12
4. **COOLDOWN State (5s adaptive):**
   - Ignore all "1" messages for minimum 5 seconds
   - **Early exit**: If "0" received, immediately return to IDLE
   - Allows fast queue processing in busy parking lots
   - Returns to IDLE after timeout

**Why Debouncing?**
Real parking involves multiple sensor fluctuations as drivers reverse, align, and adjust position. Without debouncing, one parking event could trigger 5-10 camera captures. The state machine ensures only ONE image per parking event.

**Multi-Slot Behavior:**
- Camera subscribes to wildcard: `SPARK_C06/isOccupied/*`
- Receives messages from all slots: A01, A02, A03, etc.
- Last car to stabilize within 2s window triggers capture
- Wide-angle camera captures ALL slots in single image
- Backend processes image to detect individual slot changes
- Assumption: Passengers never trigger sensors (only car entry/exit)

---

### 2. ESP32-CAM Trigger Protocol

#### Trigger Signal
```
STM32 PB12 → ESP32-CAM GPIO15 (Rising Edge Interrupt)

Timing:
┌──────────────────┐
│   IDLE (LOW)     │
└──────────────────┘
         │
         ▼
    ┌────────┐
    │  HIGH  │ ← 100ms pulse
    └────────┘
         │
         ▼
┌──────────────────┐
│   IDLE (LOW)     │
└──────────────────┘
```

**STM32 Side:**
1. Detect stable parking event (STABILIZING → CONFIRMED after 2s)
2. Assert PB12 = HIGH for 100ms
3. Return PB12 = LOW
4. Enter COOLDOWN state (5s minimum, exits early if car leaves)

**ESP32-CAM Side (Independent Operation):**
1. GPIO15 rising edge interrupt triggered
2. Wake from deep sleep
3. Initialize OV2640 camera
4. Capture VGA (640×480) grayscale image
5. JPEG compress (quality 75-80) → ~10-15 KB
6. Connect WiFi
7. HTTPS POST to backend: `https://danishritonga-spark-backend.hf.space/upload`
8. Disconnect WiFi
9. Power down camera
10. Return to deep sleep

**No SPI Data Transfer:**
- Image data never sent to STM32
- ESP32-CAM has 520KB SRAM (sufficient for VGA processing)
- ESP32-CAM handles WiFi + HTTPS (ESP-01 limitation bypassed)
- STM32 only acts as intelligent trigger controller

---

### 3. Image Upload Protocol (ESP32-CAM → Backend)

**Note:** HTTP upload is handled entirely by ESP32-CAM. STM32 is NOT involved in image transfer.

#### ESP32-CAM HTTPS POST (Independent)
```
Endpoint: https://danishritonga-spark-backend.hf.space/upload
Method: POST
Content-Type: multipart/form-data or application/octet-stream
Payload: JPEG compressed image (~10-15 KB)
```

**Image Specifications:**
- **Resolution:** VGA (640×480)
- **Format:** Grayscale
- **Compression:** JPEG (quality 75-80)
- **Size:** ~10-15 KB (compressed)
- **Encoding Time:** ~200-300ms @ 240MHz

**Upload Flow (ESP32-CAM):**
```
1. Capture VGA grayscale frame → ESP32 SRAM
2. JPEG compress using hardware encoder
3. Connect WiFi (credentials stored in ESP32 firmware)
4. HTTPS POST compressed image to backend
5. Receive HTTP 200 OK response
6. Disconnect WiFi
7. Return to deep sleep
```

**Why ESP32-CAM Handles Upload:**
- ✅ HTTPS support (ESP-01 only supports HTTP)
- ✅ 520KB SRAM (sufficient for VGA + JPEG encoding)
- ✅ Hardware JPEG encoder (fast compression)
- ✅ Native WiFi (no UART bottleneck)
- ✅ Reduces STM32 complexity (no image buffering)
- ✅ Lower power (WiFi only active during upload)

---

## State Machine

### Smart Parking Debouncing State Machine

```
┌─────────────────────────────────────────────────┐
│                    IDLE                         │
│  - No car detected (spot empty)                 │
│  - PB12 = LOW                                    │
│  - Waiting for first "1" message                │
└──────────────┬──────────────────────────────────┘
               │ MQTT: "SPARK_C06/isOccupied/*" → "1"
               ▼
┌─────────────────────────────────────────────────┐
│                STABILIZING                       │
│  - Car detected (driver aligning)               │
│  - PB12 = LOW (camera NOT triggered yet)        │
│  - Timer: 2 seconds                             │
│  - Any new "1" resets timer                     │
│  - If "0" received → back to IDLE               │
└──────────────┬──────────────────────────────────┘
               │ 2s elapsed without state change
               ▼
┌─────────────────────────────────────────────────┐
│                 CONFIRMED                        │
│  - Car stable → trigger camera                  │
│  - Assert PB12 = HIGH for 100ms                 │
│  - ESP32-CAM captures parking event             │
└──────────────┬──────────────────────────────────┘
               │ After 100ms
               ▼
┌─────────────────────────────────────────────────┐
│                 COOLDOWN                         │
│  - PB12 = LOW                                    │
│  - Ignore all "1" messages                      │
│  - Duration: 5 seconds (minimum)                │
│  - Early exit: "0" → IDLE immediately           │
│  - Allows fast queue processing                 │
└──────────────┬──────────────────────────────────┘
               │ After 5s timeout OR "0" received
               └──────► Back to IDLE (ready for next car)
```

### State Transitions

| Current State | Event | Next State | Action |
|---------------|-------|------------|--------|
| IDLE | "1" received | STABILIZING | Start 2s timer, blink LED (2×) |
| STABILIZING | "1" received | STABILIZING | Reset 2s timer, toggle LED |
| STABILIZING | "0" received | IDLE | Cancel trigger, blink LED (1×) |
| STABILIZING | 2s timeout | CONFIRMED | Assert PB12=HIGH, blink LED (3×) |
| CONFIRMED | 100ms elapsed | COOLDOWN | PB12=LOW, start 5s timer |
| COOLDOWN | "1" message | COOLDOWN | Ignored (no action) |
| COOLDOWN | "0" message | IDLE | Early exit, blink LED (2×) |
| COOLDOWN | 5s timeout | IDLE | Blink LED (2× slow), ready |

### Connection Management

**Keep-Alive:**
- MQTT PINGREQ every 30 seconds
- Connection check every 60 seconds
- Auto-reconnect on failure

**Reconnection Sequence:**
```
1. Detect disconnection
2. Close TCP connection (AT+CIPCLOSE)
3. Reconnect WiFi (AT+CWJAP)
4. Reconnect MQTT broker (AT+CIPSTART)
5. Send MQTT CONNECT packet
6. Resubscribe to command topic
```

---

## LED Status Indicators

### Initialization Patterns
| Blink Pattern | Meaning | State |
|---------------|---------|-------|
| 3 slow (200ms) | System ready | Startup |
| 3 medium (300ms) | WiFi connected | Initialization |
| 5 slow (200ms) | MQTT connected | Initialization |
| 4 fast (150ms) | Subscribed to topic | Initialization |

### Smart Parking State Machine Patterns
| Blink Pattern | Meaning | State |
|---------------|---------|-------|
| 2 fast (100ms) | "1" received → STABILIZING | Car detected |
| Toggle | Additional "1" in STABILIZING | Car still moving |
| 1 slow (300ms) | "0" received in STABILIZING | Car left (abort) |
| 3 fast (100ms) | STABILIZING → CONFIRMED | Triggering camera |
| 1 quick (50ms) | CONFIRMED → COOLDOWN | Pulse complete |
| 2 slow (200ms) | COOLDOWN → IDLE (timeout or early exit) | Ready for next car |

### Error Patterns
| Blink Pattern | Meaning | State |
|---------------|---------|-------|
| 10 medium (200ms) | WiFi connection failed | Error |
| 10 slow (300ms) | MQTT TCP failed | Error |
| 10 very slow (400ms) | MQTT CONNECT failed | Error |
| 10 rapid (50ms) | Reconnection attempt | Recovery |
| 15 fast (100ms) | Reconnection failed | Error |

---

## Memory Usage

### RAM Allocation
```
Total RAM:      65,536 bytes (64KB)
Used:           776 bytes (1.2%)
Available:      64,760 bytes (98.8%)
```

**Key Buffers:**
- UART RX Buffer: 512 bytes (ESP-01 communication)
- State Machine Variables: ~12 bytes
- Stack/Heap: ~200 bytes
- HAL/System: ~52 bytes

**Memory Savings vs v1.0:**
- No SPI chunk buffer (512 bytes freed)
- No image buffering (76,800 bytes never allocated)
- Reduced from 1,372 bytes (2.1%) to 776 bytes (1.2%)

### Flash Allocation
```
Total Flash:    262,144 bytes (256KB)
Used:           9,996 bytes (3.8%)
Available:      252,148 bytes (96.2%)
```

**Flash Savings vs v1.0:**
- Removed SPI transfer code (~1 KB)
- Removed HTTP streaming code (~1.5 KB)
- Simplified state machine logic
- Reduced from 12,756 bytes (4.9%) to 9,996 bytes (3.8%)

---

## Function Reference

### Initialization Functions

#### `SystemClock_Config()`
Configures system clock to 84MHz using HSI + PLL.
- HSI: 16MHz
- PLL_M: 16
- PLL_N: 336
- PLL_P: 4 (÷4)
- SYSCLK: 84MHz

#### `GPIO_Init()`
Initializes GPIO pins:
- PC13: LED output (push-pull)
- PB12: SPI CS output (push-pull, high speed)

#### `USART1_Init()`
Configures UART for ESP-01:
- Baud: 115200
- Data: 8-bit
- Stop: 1-bit
- Parity: None
- Interrupt: RX enabled

#### `SPI1_Init()`
Configures SPI for ESP32-CAM:
- Mode: Master
- Speed: 5.25MHz (PCLK/16)
- Data: 8-bit
- CPOL: Low
- CPHA: 1st edge

### ESP-01 Functions

#### `ESP01_ConnectWiFi(ssid, password)`
**Returns:** `1` on success, `0` on failure  
**Timeout:** 10 seconds  
Sends AT commands to connect to WiFi network.

#### `ESP01_ConnectMQTTBroker(broker, port)`
**Returns:** `1` on success, `0` on failure  
**Timeout:** 5 seconds  
Opens TCP connection to MQTT broker.

#### `ESP01_Send_MQTT_CONNECT(clientID)`
**Returns:** `1` on success, `0` on failure  
Sends MQTT CONNECT packet with clean session flag.

#### `ESP01_Send_MQTT_SUBSCRIBE(topic)`
Sends MQTT SUBSCRIBE packet with QoS 0.

#### `ESP01_ReceiveAndParse()`
Parses incoming UART data for MQTT PUBLISH packets.  
Sets `capture_triggered` flag when message received.

#### `ESP01_SendPing()`
Sends MQTT PINGREQ packet (keep-alive).

#### `ESP01_CheckConnection()`
**Returns:** `1` if connected, `0` if disconnected  
Checks TCP connection status using AT+CIPSTATUS.

#### `ESP01_Reconnect()`
Full reconnection sequence: WiFi → TCP → MQTT → Subscribe.

#### `ESP01_SendImageHTTP(chunk_buffer, image_size)`
**Returns:** `1` on success, `0` on failure  
Streams image data to HTTP server in 512-byte chunks.

### ESP32-CAM Functions

#### `ESP32CAM_TriggerCapture()`
Triggers image capture:
- CS LOW for 100ms
- Wait 2000ms for capture

#### `ESP32CAM_ReceiveImage(chunk_buffer, total_size)`
Receives and validates image size header:
- Reads 4-byte big-endian size
- Validates against expected 76,800 bytes
- Sets `*total_size` to received value or 0 on error

### Utility Functions

#### `Blink_Status(count, delay)`
Blinks LED for status indication.
- `count`: Number of blinks
- `delay`: Milliseconds per blink phase

#### `send_AT(cmd, delay_ms)`
Sends AT command and waits.

#### `wait_for_prompt()`
Waits for '>' prompt after AT+CIPSEND.  
**Timeout:** 3 seconds

---

## Configuration

### WiFi Settings
```c
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"
```

### MQTT Settings (Smart Parking)
```c
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_CLIENT_ID "SPARK_C06"
#define MQTT_TOPIC_BASE "SPARK_C06/isOccupied/"
#define MQTT_TOPIC_SUB MQTT_TOPIC_BASE "*"          // Wildcard subscription
#define MQTT_TOPIC_PUB "SPARK_C06/camera/status"
#define MQTT_KEYWORD_OCCUPIED "1"
#define MQTT_KEYWORD_EMPTY "0"
```

**Multi-Slot Monitoring:**
- Camera subscribes to `SPARK_C06/isOccupied/*`
- Receives messages from all on-slot sensors: A01, A02, B12, etc.
- One camera monitors entire parking row (wide-angle lens)
- No firmware changes needed when adding/removing slots

### State Machine Timing
```c
#define TRIGGER_PULSE_MS 100     // ESP32-CAM trigger pulse duration
#define STABILIZING_MS 2000      // Wait 2s for car to stabilize (reduced for multi-slot)
#define COOLDOWN_MS 5000         // Minimum 5s cooldown, exits early if car leaves
```

**Timing Rationale:**
- **2s stabilization**: Faster response for multi-slot scenarios, less chance of second car interfering
- **5s adaptive cooldown**: Minimum prevents rapid re-triggers, exits early when car leaves
- **Early exit benefit**: Next car can trigger immediately after previous car departs

### Image Settings (ESP32-CAM Side)
```c
// Note: These settings are in ESP32-CAM firmware, not STM32
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define IMAGE_FORMAT PIXFORMAT_GRAYSCALE
#define JPEG_QUALITY 80          // 75-80 recommended (~10-15 KB)
```

---

## Error Handling

### Validation Checks
1. **WiFi Connection:** Waits for "WIFI CONNECTED" or "OK"
2. **TCP Connection:** Waits for "CONNECT" or "OK"
3. **Image Size:** Must equal 76,800 bytes exactly
4. **SPI Transfer:** HAL_SPI_Receive returns HAL_OK
5. **Connection Status:** Periodic AT+CIPSTATUS checks

### Error Recovery
- **Connection Lost:** Auto-reconnect every 60 seconds
- **Invalid Size:** Skip HTTP transfer, wait for next trigger
- **SPI Timeout:** Break transfer loop, blink error code
- **Transfer Incomplete:** Blink error code, reconnect MQTT

### Timeout Values
- WiFi connect: 10,000ms
- TCP connect: 5,000ms
- SPI receive: 5,000ms
- UART transmit: 5,000ms
- HTTP response: 2,000ms

---

### Timing Constraints

### Critical Timing (STM32 State Machine)
- **Trigger Pulse:** 100ms HIGH on PB12 (ESP32-CAM detection)
- **Stabilizing Period:** 2,000ms (wait for car to stop moving)
- **Cooldown Period:** 5,000ms minimum (exits early if car leaves)
- **MQTT Ping:** 30,000ms (keep-alive interval)
- **Status Check:** 60,000ms (connection verification)

### ESP32-CAM Timing (Independent)
```
Trigger Detection → Wake from deep sleep: ~10ms
Camera Init → First frame ready: ~500ms
VGA Capture: ~100ms
JPEG Compression (quality 80): ~200-300ms
WiFi Connect: ~2-4 seconds
HTTPS POST (10-15 KB): ~1-2 seconds
Total per capture: ~4-7 seconds
```

**No UART Bottleneck:**
- Image never sent to STM32
- ESP32-CAM uses native WiFi (fast)
- No 115200 baud limitation
- Parallel operation: STM32 in COOLDOWN while ESP32-CAM uploads

---

## Workflow Example

### Complete Smart Parking Event

**Scenario:** Driver approaches slot A03, takes 3 alignment attempts, parks successfully. Another car arrives shortly after.

```
Time    Event (STM32 + ESP32-CAM)
----    -------------------------
0.0s    [IDLE] Camera monitoring slots A01-A05

2.5s    [MQTT] "SPARK_C06/isOccupied/A03" → "1"
        → STM32 → STABILIZING
        → LED blinks 2× (car detected)
        → Start 2s timer

3.8s    [MQTT] "SPARK_C06/isOccupied/A03" → "1"
        → STM32 → STABILIZING (reset timer)
        → LED toggle (car still moving)

4.2s    [MQTT] "SPARK_C06/isOccupied/A03" → "0"
        → STM32 → IDLE (car reversed out)
        → LED blinks 1× slow

5.0s    [MQTT] "SPARK_C06/isOccupied/A03" → "1"
        → STM32 → STABILIZING (restart)
        → LED blinks 2× (car returned)

6.1s    [MQTT] "SPARK_C06/isOccupied/A03" → "1"
        → STM32 → STABILIZING (reset timer)
        → LED toggle

8.1s    [2s Stable] STM32 → CONFIRMED
        → PB12 = HIGH (100ms pulse)
        → LED blinks 3× (triggering camera)

8.2s    [ESP32-CAM] GPIO15 rising edge detected
        → Wake from deep sleep
        → Init camera

8.7s    [ESP32-CAM] Capture VGA grayscale
        → 640×480 pixels captured

9.0s    [ESP32-CAM] JPEG compress
        → Quality 80, output ~12 KB

9.3s    [STM32] PB12 = LOW → COOLDOWN (5s minimum)
        → LED blinks 1× quick

11.0s   [ESP32-CAM] WiFi connect
        → "ParkingLot_WiFi" connected

12.5s   [ESP32-CAM] HTTPS POST
        → https://danishritonga-spark-backend.hf.space/upload
        → Upload 12 KB JPEG

13.8s   [ESP32-CAM] HTTP 200 OK
        → Disconnect WiFi
        → Power down camera
        → Deep sleep

14.0s   [MQTT] "SPARK_C06/isOccupied/A03" → "0" (car leaves)
        → STM32 → IDLE (early exit from COOLDOWN after 5s!)
        → LED blinks 2× slow
        → Ready for next car

14.5s   [MQTT] "SPARK_C06/isOccupied/A01" → "1" (new car, different slot)
        → STM32 → STABILIZING immediately ✅
        → No wait needed!
```

**Total STM32 Cycle:** ~12 seconds (6s stabilization attempts + 0.1s trigger + 5s cooldown + early exit)  
**ESP32-CAM Active Time:** ~5 seconds (capture + upload)  
**Parallel Operation:** ESP32-CAM processes independently while STM32 in COOLDOWN  
**Images Captured:** **1** (despite 4 MQTT "1" messages and 2 "0" messages)
**Next Car Ready:** Immediately after previous car departs (early exit benefit)

**Multi-Slot Example:**
```
0.0s    - Slot A01: "1" → STABILIZING (start 2s timer)
1.5s    - Slot A02: "1" → STABILIZING (reset timer - new car detected)
3.5s    - 2s elapsed from last "1" → CONFIRMED
        → Camera captures BOTH A01 and A02 in single image ✅
        → Backend processes image to detect individual slot changes
```

---

## Troubleshooting

### Common Issues

**1. WiFi won't connect (STM32/ESP-01)**
- Check SSID/password in code
- Verify ESP-01 has proper power (3.3V, min 300mA)
- Check UART connections (PB6→ESP-01 RX, PB7←ESP-01 TX)
- Look for 10 medium blinks (200ms)

**2. MQTT connection fails**
- Verify internet connectivity
- Check broker address: broker.hivemq.com
- Ensure unique client ID (SPARK_C06)
- Look for 10 slow blinks (300ms/400ms)
- Verify wildcard subscription: `SPARK_C06/isOccupied/*`

**3. Camera not triggering**
- Publish to correct topic: `SPARK_C06/isOccupied/A01` (or any slot ID)
- Payload must be `"1"` (occupied) or `"0"` (empty)
- Check for 2 fast LED blinks when "1" received
- Verify ESP-01 subscription worked (4 fast blinks at startup)
- Ensure state machine not in COOLDOWN (5s ignore period for "1" messages)
- Note: COOLDOWN exits early if "0" received

**4. Multiple rapid triggers (normal behavior!)**
- State machine is designed to handle this
- First "1" → STABILIZING (start 2s timer)
- Additional "1" → resets timer (car still moving)
- Camera only triggers after 2s stability
- Check LED toggles during stabilization
- This is expected for real parking scenarios

**5. Car leaves before capture**
- State machine design: "0" during STABILIZING → back to IDLE
- This is correct behavior (no wasted image)
- Check for 1 slow LED blink when car leaves early

**6. Trigger works but no image uploaded**
- Issue is on ESP32-CAM side (not STM32)
- Check ESP32-CAM firmware running
- Verify GPIO15 interrupt configured
- Check WiFi credentials on ESP32-CAM
- Verify backend endpoint: `https://danishritonga-spark-backend.hf.space/upload`
- Monitor ESP32-CAM serial output for debug info

**7. System stuck in COOLDOWN**
- Wait minimum 5 seconds after trigger
- Publish "0" (empty) to force early exit to IDLE
- Check for 2 slow blinks when returning to IDLE
- If stuck: power cycle STM32
- Verify HAL_GetTick() incrementing correctly

**8. Camera triggering too frequently**
- Check STABILIZING_MS setting (should be 2000ms)
- Verify COOLDOWN_MS minimum (should be 5000ms)
- Ensure sensors aren't sending false "1" messages
- Check if multiple slots reporting simultaneously (expected for multi-slot)

**9. Slow response to next car**
- System should exit COOLDOWN early when car leaves
- Verify "0" messages being sent by sensors
- Check for 2 blink pattern on early COOLDOWN exit
- If not exiting early, check ESP01_ReceiveAndParse() logic

### Debug via LED
Monitor LED patterns to identify system state:
- Count number of blinks
- Measure blink duration
- Reference LED Status Indicators table

---

## Future Enhancements

### Potential STM32 Improvements
1. **State Machine Tuning:**
   - Configurable timings via MQTT (STABILIZING_MS, COOLDOWN_MS)
   - Adaptive stabilization (learn parking patterns)
   - Different timing profiles for different slot types

2. **Multi-Camera Coordination:**
   - Support multiple ESP32-CAM units per STM32
   - Zone-based triggering (trigger only relevant camera)
   - Round-robin scheduling for simultaneous events

3. **Advanced Debouncing:**
   - Machine learning-based pattern recognition
   - Differentiate parking vs. passing traffic
   - Predict final parking position

4. **Power Management:**
   - STM32 sleep mode between MQTT messages
   - Dynamic clock scaling (84MHz → 16MHz during IDLE)
   - Wake-on-UART for ESP-01 messages

5. **Diagnostics:**
   - Publish state machine metrics via MQTT
   - Track: triggers/hour, false positives, average stabilization time
   - Remote configuration updates

### ESP32-CAM Enhancements (Separate Firmware)
1. **Computer Vision:**
   - On-device license plate detection
   - Vehicle counting in frame
   - Identify specific changed slot (A01 vs A02)
   - Motion detection for validation

2. **Compression:**
   - Adaptive JPEG quality based on scene complexity
   - Multi-resolution capture (thumbnail + full)
   - Edge-detected ROI extraction

3. **Reliability:**
   - Local SD card backup (if WiFi fails)
   - Retry logic with exponential backoff
   - Offline queuing (upload when WiFi returns)

4. **Smart Capture:**
   - Wait for motion to stop (accelerometer)
   - Multiple angles (if servo-mounted)
   - HDR capture for varying lighting

---

## Version History

### v2.0 (2025-11-12) - Smart Parking State Machine
- **Major architectural change:** STM32 = debouncing controller, ESP32-CAM = independent processor
- 4-state machine: IDLE → STABILIZING → CONFIRMED → COOLDOWN
- Intelligent debouncing for real-world parking scenarios
- Wildcard MQTT subscription: `SPARK_C06/isOccupied/*`
- Simple binary payloads: "1" (occupied) / "0" (empty)
- 100ms trigger pulse on PB12 → ESP32-CAM GPIO15
- ESP32-CAM handles VGA capture + JPEG + WiFi + HTTPS independently
- **Removed:** SPI data transfer, HTTP upload code from STM32
- **Memory savings:** 1.2% RAM (776 bytes), 3.8% Flash (9,996 bytes)
- Multi-slot monitoring with single camera unit

### v1.0 (2025-11-09) - Initial Release [DEPRECATED]
- MQTT trigger via ESP-01
- ESP32-CAM image capture via SPI
- HTTP POST streaming through STM32 (raw grayscale)
- Auto-reconnection
- LED status indicators
- 2.1% RAM usage (1,372 bytes)
- 4.9% Flash usage (12,756 bytes)
- **Limitations:** ESP-01 no HTTPS, STM32 buffering bottleneck, simple trigger

---

## License & Credits

**Developed for:** SPARK Project  
**Platform:** STM32F401CCU6 Black Pill  
**Date:** November 2025  

**Dependencies:**
- STM32 HAL Library (ST Microelectronics)
- CMSIS (ARM)

---

## Contact & Support

For issues or questions about this firmware:
- Check LED blink patterns for diagnostics
- Verify all hardware connections
- Review configuration defines
- Test components individually (WiFi → MQTT → Camera → HTTP)

**Hardware Requirements:**
- STM32F401CCU6 Black Pill
- ESP-01 WiFi module (MQTT gateway)
- ESP32-CAM AI-Thinker (camera + WiFi + processing)
- On-slot occupancy sensors (ultrasonic/magnetic)
- 3.3V Li-ion battery (e.g., 18650 ≥2000 mAh)
- Fixed 5V booster (MT3608)
- 3.3V regulator (AMS1117-3.3)
- Power capacitors (470µF + 100µF + 0.1µF)
- USB-Serial adapter (for programming)

**System Overview:**
```
On-Slot Sensors → MQTT → ESP-01 → STM32 (State Machine) → ESP32-CAM → Backend
                                      ↓
                            Debouncing + Trigger Logic
```

---

*End of Documentation*

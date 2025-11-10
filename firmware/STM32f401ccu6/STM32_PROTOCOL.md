# STM32F401CCU6 Firmware Protocol Documentation

## Overview
This firmware implements an IoT image capture system using STM32F401CCU6 (Black Pill) as the main controller, coordinating between ESP-01 (WiFi), ESP32-CAM (camera), and a remote HTTP server.

**Hardware:** STM32F401CCU6 (64KB RAM, 256KB Flash, 84MHz)  
**Firmware Version:** 1.0  
**Date:** November 9, 2025

---

## System Architecture

```
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│   ESP-01    │◄──────►│   STM32F4   │◄──────►│ ESP32-CAM   │
│   (WiFi)    │  UART  │  (Controller)│   SPI  │  (Camera)   │
└─────────────┘ 115200 └─────────────┘  ~1MHz └─────────────┘
      │                       │
      │ MQTT/HTTP             │ LED Status
      ▼                       ▼
┌─────────────┐        ┌─────────────┐
│   Broker    │        │   PC13 LED  │
│ HiveMQ.com  │        │  Indicator  │
└─────────────┘        └─────────────┘
```

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

### ESP32-CAM (Camera Module) - SPI Interface
| STM32 Pin | Function | ESP32-CAM Pin | Description |
|-----------|----------|---------------|-------------|
| PB12 | SPI1_NSS (CS) | GPIO15 | Chip Select / Trigger |
| PB13 | SPI1_SCK | GPIO14 | SPI Clock |
| PB14 | SPI1_MISO | GPIO12 | Data from Camera |
| PB15 | SPI1_MOSI | GPIO13 | Data to Camera |
| GND | Ground | GND | Common ground |

**SPI Speed:** ~5.25 MHz (84MHz / 16)  
**SPI Mode:** Mode 0 (CPOL=0, CPHA=0)  
**CS Logic:** Active LOW

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

#### MQTT Topics
- **Subscribe:** `SPARK_C06/stm32/command`
- **Client ID:** `SPARK_C06`
- **Broker:** `broker.hivemq.com:1883`

#### Trigger Mechanism
Any MQTT PUBLISH message received on the subscribed topic triggers image capture.

---

### 2. ESP32-CAM SPI Protocol

#### Trigger Sequence
```
1. CS LOW (100ms pulse) → Trigger image capture
2. Wait 2000ms → Allow camera to capture and prepare
3. CS LOW → Begin SPI data transfer
```

#### Data Transfer Protocol

**Phase 1: Size Header (4 bytes)**
```c
Byte 0: Size MSB (bits 31-24)
Byte 1: Size (bits 23-16)
Byte 2: Size (bits 15-8)
Byte 3: Size LSB (bits 7-0)

Example for 76,800 bytes (QVGA):
[0x00][0x01][0x2C][0x00]
```

**Phase 2: Image Data (76,800 bytes)**
```
Format: Raw grayscale pixels
Resolution: 320×240 (QVGA)
Pixel Format: 8-bit grayscale (0-255)
Total Size: 76,800 bytes
```

#### SPI Timing
```
Trigger:     CS LOW (100ms) → CS HIGH
Wait:        2000ms
Header Rx:   CS LOW → Read 4 bytes → CS HIGH
Data Rx:     CS LOW → Read 76,800 bytes (512-byte chunks)
```

---

### 3. HTTP Protocol (via ESP-01)

#### Connection Flow
```
1. Close existing connections
   AT+CIPCLOSE

2. Open TCP connection
   AT+CIPSTART="TCP","<server>",80

3. Send data size
   AT+CIPSEND=<total_size>
   
4. Wait for '>' prompt

5. Send HTTP request + image data
```

#### HTTP POST Request Format
```http
POST /api/upload HTTP/1.1
Host: <server>
Content-Type: application/octet-stream
X-Image-Width: 320
X-Image-Height: 240
X-Image-Format: grayscale
Content-Length: 76800
Connection: close

<binary_image_data>
```

#### Custom Headers
- `X-Image-Width`: Image width in pixels (320)
- `X-Image-Height`: Image height in pixels (240)
- `X-Image-Format`: Pixel format (grayscale)

#### Streaming Architecture
```
ESP32-CAM (SPI) → STM32 (512-byte chunks) → ESP-01 (UART) → HTTP Server
                   ↑ Zero-copy streaming ↑
```

**Chunk Size:** 512 bytes  
**Total Chunks:** 150 (76,800 / 512)  
**Transmission Time:** ~6.6 seconds @ 115200 baud

---

## State Machine

### Main Loop States

```
┌─────────────┐
│   IDLE      │ ← Waiting for MQTT message
└──────┬──────┘
       │ MQTT PUBLISH received
       ▼
┌─────────────┐
│  TRIGGERED  │ ← Set capture_triggered flag
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  TRIGGER    │ ← CS LOW pulse to ESP32-CAM
│   CAMERA    │   Wait 2000ms
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RECEIVE    │ ← Read 4-byte header
│   HEADER    │   Validate size
└──────┬──────┘
       │ size == 76,800
       ▼
┌─────────────┐
│  STREAM     │ ← Open HTTP connection
│   IMAGE     │   Stream 512-byte chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RECONNECT  │ ← Close HTTP connection
│    MQTT     │   Reconnect to MQTT broker
└──────┬──────┘
       │
       └──────► Back to IDLE
```

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

| Blink Pattern | Meaning | State |
|---------------|---------|-------|
| 3 slow (200ms) | System ready | Startup |
| 3 medium (300ms) | WiFi connected | Initialization |
| 5 slow (200ms) | MQTT connected | Initialization |
| 4 fast (150ms) | Subscribed to topic | Initialization |
| 5 rapid (50ms) | MQTT trigger received | Active |
| 2 slow (200ms) | Valid image size | Capture |
| Toggle during transfer | Image streaming | Active |
| 5 slow (300ms) | HTTP success | Success |
| 8 fast (150ms) | Transfer failed | Error |
| 10 fast (100ms) | Invalid image size | Error |
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
Used:           1,372 bytes (2.1%)
Available:      64,164 bytes (97.9%)
```

**Key Buffers:**
- UART RX Buffer: 512 bytes
- SPI Chunk Buffer: 512 bytes
- Stack/Heap: ~300 bytes
- HAL/System: ~48 bytes

### Flash Allocation
```
Total Flash:    262,144 bytes (256KB)
Used:           12,756 bytes (4.9%)
Available:      249,388 bytes (95.1%)
```

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

### MQTT Settings
```c
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_CLIENT_ID "SPARK_C06"
#define MQTT_TOPIC_SUB "SPARK_C06/stm32/command"
```

### HTTP Settings
```c
#define HTTP_SERVER "your-server.com"     // ⚠️ Update before deployment
#define HTTP_PORT 80
#define HTTP_ENDPOINT "/api/upload"
```

### Image Settings
```c
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240
#define IMAGE_SIZE 76800  // QVGA grayscale
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

## Timing Constraints

### Critical Timing
- **CS Trigger Pulse:** 100ms (minimum for ESP32-CAM recognition)
- **Capture Wait:** 2000ms (ESP32-CAM processing time)
- **Chunk Delay:** 20ms (between 512-byte SPI chunks)
- **MQTT Ping:** 30,000ms (keep-alive interval)
- **Status Check:** 60,000ms (connection verification)

### Transmission Time Estimates
```
UART @ 115200 baud:
- 1 byte = 86.8μs
- 512 bytes = 44.4ms
- 76,800 bytes = 6.67 seconds (theoretical)
- 76,800 bytes ≈ 9.6 seconds (with delays)
```

---

## Workflow Example

### Complete Capture Cycle

```
Time    Event
----    -----
0.0s    MQTT PUBLISH received
        → LED blinks 5 times rapidly
        → capture_triggered = 1

0.3s    ESP32CAM_TriggerCapture()
        → CS LOW (100ms)
        → CS HIGH
        
2.3s    ESP32CAM_ReceiveImage()
        → CS LOW
        → Read 4-byte header
        → Validate: 0x00 0x01 0x2C 0x00 = 76,800 ✓
        → CS HIGH
        → LED blinks 2 times (valid size)
        
2.8s    ESP01_SendImageHTTP()
        → AT+CIPCLOSE
        → AT+CIPSTART="TCP","server.com",80
        → AT+CIPSEND=<size>
        → Wait for '>'
        
8.3s    HTTP header sent
        → POST /api/upload HTTP/1.1
        → Custom headers
        
8.4s    Image streaming begins
        → CS LOW
        → Loop 150 iterations:
          - SPI receive 512 bytes
          - UART transmit 512 bytes
          - LED toggle
          - Delay 20ms
        → CS HIGH
        
18.4s   Wait for HTTP response (2s)

20.4s   Reconnect MQTT
        → AT+CIPCLOSE
        → AT+CIPSTART (broker)
        → MQTT CONNECT
        → MQTT SUBSCRIBE
        
22.4s   Back to IDLE
        → Ready for next trigger
```

**Total Cycle Time:** ~22-25 seconds

---

## Troubleshooting

### Common Issues

**1. WiFi won't connect**
- Check SSID/password in code
- Verify ESP-01 has proper power (3.3V, min 300mA)
- Check UART connections (TX↔RX, RX↔TX)
- Look for 10 medium blinks (200ms)

**2. MQTT connection fails**
- Verify internet connectivity
- Check broker address: broker.hivemq.com
- Ensure unique client ID
- Look for 10 slow blinks (300ms/400ms)

**3. Image capture not triggering**
- Publish to correct topic: `SPARK_C06/stm32/command`
- Check for 5 rapid LED blinks when message arrives
- Verify ESP-01 subscription worked (4 fast blinks)

**4. Invalid image size**
- Check ESP32-CAM firmware sends size header correctly
- Verify big-endian byte order
- Look for 10 fast blinks (100ms)
- Expected: [0x00][0x01][0x2C][0x00]

**5. SPI communication fails**
- Check SPI connections (MISO/MOSI)
- Verify CS pin controls ESP32-CAM
- Ensure ESP32-CAM firmware running
- Look for 8 fast blinks (150ms)

**6. HTTP upload fails**
- Update HTTP_SERVER in code
- Verify server accepts raw binary POST
- Check Content-Type handling on server
- Look for 8 fast blinks (150ms)

### Debug via LED
Monitor LED patterns to identify system state:
- Count number of blinks
- Measure blink duration
- Reference LED Status Indicators table

---

## Future Enhancements

### Potential Improvements
1. **Image Processing:**
   - 2×2 downsampling (160×120) for faster transmission
   - ROI extraction (center crop)
   - Adaptive resolution based on bandwidth

2. **Compression:**
   - JPEG compression on ESP32-CAM (hardware encoder)
   - Streaming JPEG compression (8×8 blocks)
   - Run-length encoding for simple scenes

3. **Protocol:**
   - HTTPS support (if ESP-01 firmware allows)
   - MQTT QoS 1 with acknowledgment
   - Multi-part upload for reliability

4. **Power Management:**
   - Deep sleep between captures
   - Dynamic frequency scaling
   - Peripheral power gating

5. **Features:**
   - Multiple image formats (RGB565, JPEG)
   - Configurable resolution via MQTT
   - Image metadata (timestamp, sequence number)
   - Local SD card storage (backup)

---

## Version History

### v1.0 (2025-11-09)
- Initial release
- MQTT trigger via ESP-01
- ESP32-CAM image capture via SPI
- HTTP POST streaming (raw grayscale)
- Auto-reconnection
- LED status indicators
- 2.1% RAM usage (1,372 bytes)
- 4.9% Flash usage (12,756 bytes)

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
- ESP-01 WiFi module
- ESP32-CAM (AI-Thinker)
- 3.3V power supply (min 500mA)
- USB-Serial adapter (for programming)

---

*End of Documentation*

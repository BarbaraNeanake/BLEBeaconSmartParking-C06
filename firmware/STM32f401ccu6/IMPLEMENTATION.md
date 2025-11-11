# STM32F401 + ESP32-CAM + ESP-01 Implementation

## Overview
This firmware implements an MQTT-triggered camera system that:
1. Connects ESP-01 to WiFi and MQTT broker (HiveMQ)
2. Subscribes to MQTT command topic
3. On MQTT message, triggers ESP32-CAM to capture grayscale VGA image
4. Receives pre-compressed JPEG via SPI from ESP32-CAM
5. Streams image data directly to REST API server via HTTP POST

## Hardware Configuration

### Pin Mappings (from blueprint.md)
- **PC13**: LED indicator
- **PB6**: USART1_TX (to ESP-01 RX)
- **PB7**: USART1_RX (from ESP-01 TX)
- **PB12**: SPI1_NSS/CS (ESP32-CAM trigger)
- **PB13**: SPI1_SCK
- **PB14**: SPI1_MISO
- **PB15**: SPI1_MOSI

### System Clock
- **84MHz** via PLL (HSI 16MHz × 21 / 4)
- APB1: 42MHz
- APB2: 84MHz
- Flash latency: 2 wait states

## Network Configuration

### WiFi (ESP-01)
```c
SSID: "Waifai"
Password: "123654987"
Baud Rate: 115200
```

### MQTT (HiveMQ Public Broker)
```c
Broker: broker.hivemq.com:1883
Client ID: "SPARK_C06"
Subscribe Topic: "SPARK_C06/stm32/command"
QoS: 0
Keep-alive: 30 seconds (PINGREQ)
```

### HTTP REST API
```c
Server: "your-server.com"  // ⚠️ UPDATE THIS
Port: 80
Endpoint: "/api/upload"
Content-Type: image/jpeg
```

## Memory Architecture

### RAM Usage: ~2.1% (1368 bytes / 64KB)
The STM32F401CC has only 64KB RAM, which cannot fit a full 640×480 grayscale image (307KB).

**Solution: Streaming Architecture**
- ESP32-CAM captures and compresses image to JPEG (~10-15KB)
- STM32 receives via SPI in 512-byte chunks
- Chunks streamed directly to HTTP server via ESP-01 UART
- No full image buffer needed

### Flash Usage: ~4.8% (12608 bytes / 256KB)

## Image Pipeline

### Workflow
```
MQTT Trigger → STM32 Pulses CS Pin → ESP32-CAM Captures
    ↓
ESP32-CAM Compresses to JPEG (grayscale VGA, quality ~75)
    ↓
Sends 4-byte size header + JPEG data via SPI
    ↓
STM32 opens HTTP connection via ESP-01
    ↓
Streams JPEG chunks: SPI → STM32 → UART → ESP-01 → HTTP Server
    ↓
Closes HTTP, reconnects MQTT
```

### Expected Image Specifications (from blueprint)
- **Format**: Grayscale VGA (640×480)
- **Uncompressed size**: 307,200 bytes
- **JPEG compressed**: ~10-15KB @ quality 75-80
- **Capture time**: ~100ms (ESP32-CAM)
- **Transfer time**: ~2-3s (SPI + UART @ 115.2k baud)

## ESP32-CAM Firmware Requirements

⚠️ **IMPORTANT**: The ESP32-CAM must be programmed to:
1. Remain in SPI slave mode
2. Wait for CS pin LOW pulse (trigger)
3. Capture grayscale VGA image
4. Compress to JPEG (quality 75-80)
5. Send via SPI:
   - 4-byte header: image size (big-endian)
   - JPEG data bytes

### Example ESP32-CAM SPI Response Format
```
[SIZE_MSB] [SIZE_2] [SIZE_3] [SIZE_LSB] [JPEG_DATA...]
Example: 0x00 0x00 0x2D 0x3A → 11,578 bytes
```

## Status LED Indicators

| Pattern | Meaning |
|---------|---------|
| 3 blinks (200ms) | WiFi connected |
| 5 blinks (200ms) | MQTT connected |
| 4 blinks (150ms) | Subscribed to topic |
| 5 rapid blinks (50ms) | MQTT message received |
| 2 blinks (200ms) | Image size validated |
| Toggle during transfer | Data streaming |
| 5 blinks (300ms) | HTTP POST successful |
| 8 blinks (150ms) | HTTP POST failed |
| 10 blinks (100ms) | Error occurred |
| 10 rapid blinks (50ms) | Reconnection attempt |
| 15 blinks (100ms) | Reconnection failed |

## Configuration Steps

### 1. Update HTTP Server Settings
Edit `main.c`:
```c
#define HTTP_SERVER "your-server.com"  // Your API server
#define HTTP_PORT 80
#define HTTP_ENDPOINT "/api/upload"
```

### 2. Update WiFi Credentials (if needed)
```c
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"
```

### 3. Build and Upload
```bash
platformio run --target upload
```

### 4. Monitor Serial Output
```bash
platformio device monitor --baud 115200
```

## Testing

### 1. Test MQTT Connection
Publish to topic using any MQTT client:
```bash
mosquitto_pub -h broker.hivemq.com -t "SPARK_C06/stm32/command" -m "capture"
```

### 2. Expected Behavior
1. LED blinks 5 times rapidly (message received)
2. LED blinks 2 times (image size OK)
3. LED toggles during transfer
4. LED blinks 5 times (upload successful)

### 3. Check Server
Your HTTP server at `/api/upload` should receive JPEG image with:
- Content-Type: image/jpeg
- Size: ~10-15KB
- Resolution: 640×480 grayscale

## JPEG Encoder Library

A lightweight JPEG encoder is included in `lib/jpeg_encoder/`:
- **Note**: This is a simplified implementation
- **Status**: Currently unused (ESP32-CAM does compression)
- **Reason**: STM32F401's 64KB RAM cannot hold full uncompressed image
- **Future**: Could be used with streaming DCT if image arrives uncompressed

If you need STM32-side compression:
- Use external SRAM/SDRAM
- Or implement streaming DCT encoder
- Or use STM32H7 with larger RAM

## Reconnection & Reliability

### Automatic Reconnection
- **Keep-alive pings**: Every 30 seconds
- **Connection check**: Every 60 seconds
- **On disconnect**: Full WiFi + MQTT reconnect sequence

### Interrupt-Based Reception
- UART reception uses interrupts (HAL_UART_Receive_IT)
- Continuous byte accumulation in circular buffer
- Reliable detection of +IPD messages

## Known Limitations

1. **RAM Constraint**: Cannot store full uncompressed image
   - **Mitigation**: ESP32-CAM pre-compresses

2. **UART Speed**: 115.2k baud limits transfer speed
   - **15KB image**: ~1.3 seconds at max throughput
   - **Actual**: ~2-3 seconds with protocol overhead

3. **Single HTTP Connection**: MQTT disconnects during image upload
   - **Mitigation**: Auto-reconnects after HTTP transfer

4. **No Error Recovery**: If SPI transfer fails, must retry manually
   - **Future**: Implement retry logic

## Troubleshooting

### "WiFi Connected" but no MQTT blinks
- Check broker is accessible: `ping broker.hivemq.com`
- Verify port 1883 not blocked by firewall

### "MQTT Connected" but no image capture
- Check ESP32-CAM SPI firmware is running
- Verify SPI wiring: PB12-15
- Measure CS pin signal with scope

### Image transfer starts but fails
- Check HTTP server is reachable
- Verify server accepts large POST bodies
- Monitor ESP-01 AT responses
- Ensure stable power supply (ESP-01 needs stable 3.3V @ 300mA)

### Random disconnections
- Add decoupling capacitors as per blueprint
- Check power supply stability under load
- Verify UART connections (loose wires cause missed bytes)

## Performance Metrics

### Timing (Typical)
- WiFi connect: ~8-10 seconds
- MQTT connect: ~3 seconds
- Image capture: ~100ms
- SPI transfer (15KB): ~1 second @ 1MHz SPI
- HTTP POST: ~2-3 seconds total

### Power Consumption
- Idle (MQTT connected): ~150mA @ 3.3V
- Image capture: ~300mA @ 3.3V (ESP32-CAM active)
- HTTP transfer: ~200mA @ 3.3V

## Future Enhancements

1. **Add SD card support** for offline storage
2. **Implement retry logic** for failed transfers
3. **Add MQTT publish** to report status
4. **OTA updates** via ESP-01
5. **Power optimization** (deep sleep between captures)
6. **Authentication** for HTTP API (Bearer token)
7. **HTTPS support** (if ESP-01 AT firmware supports)
8. **Image metadata** (timestamp, sequence number)

## References

- Blueprint: `blueprint.md`
- PlatformIO Config: `platformio.ini`
- Main Firmware: `src/main.c`
- JPEG Encoder: `lib/jpeg_encoder/`

## Version History

- **v1.0** (November 8, 2025): Initial implementation with streaming architecture
  - MQTT trigger working
  - SPI image reception
  - HTTP POST streaming
  - 84MHz system clock
  - Auto-reconnection

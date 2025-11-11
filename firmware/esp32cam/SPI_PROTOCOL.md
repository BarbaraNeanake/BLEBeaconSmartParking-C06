# ESP32-CAM ↔ STM32 SPI Communication Protocol

## Overview
ESP32-CAM operates as **SPI Slave**, STM32F401 operates as **SPI Master**.

### Hardware Connections (as per blueprint.md)
| ESP32-CAM GPIO | STM32 Pin | Function |
|----------------|-----------|----------|
| GPIO12         | PB13      | SCLK (SPI Clock) |
| GPIO13         | PB15      | MOSI (Master Out Slave In) |
| GPIO14         | PB14      | MISO (Master In Slave Out) |
| GPIO15         | PB12      | CS/NSS (Chip Select / Trigger) |

### SPI Configuration
- **Mode**: SPI Mode 0 (CPOL=0, CPHA=0)
- **Max Transfer Size**: 4096 bytes per transaction
- **Image Format**: Grayscale
- **Image Size**: QVGA (320x240) = 76,800 bytes

⚠️ **Updated to match STM32 implementation**

---

## Communication Protocol

⚠️ **UPDATED - Now matches STM32 implementation exactly**

### Trigger and Transfer Sequence

**Phase 1: Trigger Capture**
```
STM32: CS LOW (100ms pulse) → triggers interrupt on ESP32-CAM
ESP32: Detects falling edge on GPIO15, captures image
Wait: 2000ms for capture to complete
```

**Phase 2: Read Size Header**
```
STM32: CS LOW → begins SPI transaction
ESP32: Sends 4 bytes (big-endian image size)
  Byte 0: (size >> 24) & 0xFF
  Byte 1: (size >> 16) & 0xFF
  Byte 2: (size >> 8) & 0xFF
  Byte 3: size & 0xFF
STM32: CS HIGH → ends transaction
```

**Phase 3: Read Image Data**
```
STM32: CS LOW → begins SPI transaction
ESP32: Streams 76,800 bytes in 512-byte chunks
STM32: Receives all data
STM32: CS HIGH → ends transaction
ESP32: Auto-releases frame buffer
```

---

## Complete Capture Sequence Example

### Step 1: Trigger Capture (100ms CS pulse)
```c
// STM32 triggers capture with CS LOW pulse
HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_RESET);
HAL_Delay(100);  // 100ms pulse
HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_SET);

// Wait for ESP32-CAM to capture image
HAL_Delay(2000);  // 2 second wait
```

### Step 2: Read 4-Byte Size Header
```c
uint8_t header[4];
HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_RESET);
HAL_SPI_Receive(&hspi1, header, 4, HAL_MAX_DELAY);
HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_SET);

// Parse size (big-endian)
uint32_t image_size = (header[0] << 24) | (header[1] << 16) | 
                      (header[2] << 8) | header[3];

// Expected: 0x00 0x01 0x2C 0x00 = 76,800 bytes
if (image_size == 76800) {
    printf("Valid size: %lu bytes\n", image_size);
} else {
    printf("ERROR: Invalid size %lu\n", image_size);
}
```

### Step 3: Read Image Data in 512-byte Chunks
```c
uint8_t chunk[512];
uint32_t offset = 0;

// Short delay before data transfer
HAL_Delay(10);

HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_RESET);

while (offset < image_size) {
    HAL_SPI_Receive(&hspi1, chunk, 512, HAL_MAX_DELAY);
    
    // Process chunk (compress, transmit via UART, etc.)
    ProcessImageChunk(chunk, 512);
    
    offset += 512;
    HAL_Delay(5);  // Small delay between chunks
}

HAL_GPIO_WritePin(SPI_CS_GPIO_Port, SPI_CS_Pin, GPIO_PIN_SET);

printf("Transfer complete\n");
```

---

## Timing Considerations

### Camera Capture Time
- OV2640 QVGA grayscale: ~200-300ms

### SPI Transfer Time
- SPI Clock: ~5.25 MHz (STM32F401 @ 84MHz / 16)
- 512-byte chunk: ~0.78ms @ 5.25MHz
- Full 76.8KB image: 150 chunks × 0.78ms = **~117ms transfer time**

### Total Sequence Time
- Trigger pulse: 100ms
- Capture wait: 2000ms
- Header read: <1ms
- Data transfer: 117ms
- **Total: ~2.2 seconds per image**

---

## Error Handling

### ESP32-CAM Error Responses
- `0xFF`: Camera capture failed
- No response: Check SPI wiring, CS pin, clock signal

### STM32 Error Handling
```c
// Timeout on SPI transactions
if (HAL_SPI_Transmit(...) != HAL_OK) {
    // Retry or report error
}

// Invalid response header
if (resp[0] != 0xAA && resp[0] != 0xBB) {
    // Communication error
}

// Image size validation
if (image_size > 320000) {
    // Unexpected size - possible corruption
}
```

---

## Memory Management

### ESP32-CAM
- Frame buffer automatically released after transfer
- Only 1 frame buffer in DRAM (~76.8KB)
- Ready for next capture after transfer completes

### STM32F401
- 64KB SRAM total
- Cannot store full image in RAM
- **Must stream on-the-fly** as chunks arrive
- Process each 512-byte chunk: compress → transmit → discard
- Stream to UART/ESP-01 for HTTP upload

---

## Power Optimization

- ESP32-CAM in deep sleep between captures
- Wake on GPIO15 (CS) interrupt
- After release command, return to sleep
- Current: ~80mA active, ~10µA deep sleep

---

## Blueprint Integration

This protocol matches the blueprint specifications:
- ✅ STM32 triggers ESP32-CAM via SPI CS (GPIO15)
- ✅ Captures VGA grayscale (640x480)
- ✅ Transfers via SPI in 4KB chunks
- ✅ STM32 compresses image (implement JPEG on STM32 side)
- ✅ Publishes via ESP-01 REST API (STM32 handles HTTP over UART to ESP-01)

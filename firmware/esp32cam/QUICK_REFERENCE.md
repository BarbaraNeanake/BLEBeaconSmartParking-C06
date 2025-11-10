# ESP32-CAM Trigger-Based Operation - Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                         SYSTEM FLOW                              │
└─────────────────────────────────────────────────────────────────┘

                          STM32F401
                         (SPI Master)
                              │
                              │ SPI CS Low (GPIO15)
                              │ Send 0x01 Command
                              ▼
                         ESP32-CAM
                        (SPI Slave)
                              │
                              │ Trigger Interrupt
                              │ Capture VGA Grayscale
                              ▼
                        OV2640 Sensor
                              │
                              │ 640x480 pixels
                              │ ~307,200 bytes
                              ▼
                         Frame Buffer
                         (DRAM Storage)
                              │
                              │ Return size/dimensions
                              │ Response: 0xAA + metadata
                              ▼
                          STM32F401
                              │
                              │ Request chunks (0x02)
                              │ Loop: 4KB at a time
                              │ 75 transactions
                              ▼
                      JPEG Compression
                      (libjpeg-turbo)
                              │
                              │ Quality 75-80
                              │ ~10-12 KB output
                              ▼
                           ESP-01
                         (UART TX)
                              │
                              │ AT Commands
                              │ HTTP POST
                              ▼
                         REST API
                        (Cloud Server)


┌─────────────────────────────────────────────────────────────────┐
│                         PIN MAPPING                              │
└─────────────────────────────────────────────────────────────────┘

    ESP32-CAM          Wire Color       STM32 PB12-15
    ─────────────────────────────────────────────────
    GPIO12 (SCLK)  ───── Orange ─────── PB13
    GPIO13 (MOSI)  ───── Purple ─────── PB15
    GPIO14 (MISO)  ───── Gray   ─────── PB14
    GPIO15 (CS)    ───── Yellow ─────── PB12 (Trigger)
    
    5V Power       ───── Red    ─────── 5V Booster
    GND            ───── Black  ─────── GND


┌─────────────────────────────────────────────────────────────────┐
│                    COMMAND REFERENCE                             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────┬────────────────────────────────┐
│   Trigger    │  Signal      │         Description            │
├──────────────┼──────────────┼────────────────────────────────┤
│   CAPTURE    │  CS LOW 100ms│ Trigger camera capture         │
│   HEADER     │  CS LOW/HIGH │ Read 4-byte size (big-endian)  │
│   DATA       │  CS LOW/HIGH │ Stream 76,800 bytes (512/chunk)│
└──────────────┴──────────────┴────────────────────────────────┘

┌──────────────┬──────────────┬────────────────────────────────┐
│   Response   │  Format      │         Description            │
├──────────────┼──────────────┼────────────────────────────────┤
│   SIZE       │  4 bytes BE  │ Image size: 0x00012C00 (76800) │
│   DATA       │  76800 bytes │ Raw grayscale QVGA (320x240)   │
└──────────────┴──────────────┴────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    TYPICAL SEQUENCE                              │
└─────────────────────────────────────────────────────────────────┘

1. STM32 → CS LOW (100ms pulse) → ESP32    (Trigger capture)
   ↓
2. Wait 2000ms                              (ESP32 captures image)
   ↓
3. STM32 → CS LOW → Read 4 bytes → CS HIGH (Get size: 76,800)
   ↓
4. STM32 → CS LOW → Read 76,800 bytes      (Stream in 512-byte chunks)
   ESP32 → Stream data → STM32
   STM32 → Process/compress → Buffer
   ↓
5. STM32 → CS HIGH                          (ESP32 auto-releases)
   ↓
6. STM32 → JPEG to ESP-01 → REST API


┌─────────────────────────────────────────────────────────────────┐
│                    TIMING BUDGET                                 │
└─────────────────────────────────────────────────────────────────┘

CS Trigger Pulse:    100 ms
Camera Capture:      2000 ms (wait time)
SPI Header:          <1 ms   (4 bytes)
SPI Transfer:        117 ms  (150 chunks × 0.78ms @ 5.25MHz SPI)
JPEG Compression:    500 ms  (on STM32, ~10KB output)
ESP-01 Upload:       ~2 sec  (115200 baud UART + HTTP)
                     ───────
TOTAL:               ~4.7 seconds per image


┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY FOOTPRINT                              │
└─────────────────────────────────────────────────────────────────┘

ESP32-CAM:
  - Frame buffer:  76,800 bytes (DRAM)
  - SPI buffers:    4,096 bytes
  - Total:         ~81 KB

STM32F401 (64KB SRAM):
  - Input chunk:       512 bytes
  - JPEG encoder:   12,000 bytes (workspace)
  - Output buffer:  16,000 bytes (compressed data)
  - UART buffer:       512 bytes
  - Total:          ~29 KB (fits in SRAM!)


┌─────────────────────────────────────────────────────────────────┐
│                    BUILD COMMAND                                 │
└─────────────────────────────────────────────────────────────────┘

# PlatformIO
platformio run --target upload --target monitor

# Or use VS Code PlatformIO extension:
# Click "Upload and Monitor" button

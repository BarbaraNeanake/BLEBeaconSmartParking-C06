# ESP32-CAM + STM32F401 + ESP-01 Camera Unit

> **Minimal, stable, trigger-based PoC with debouncing**  
> ESP-01 receives MQTT → STM32 acts as debounced state machine → triggers ESP32-CAM → ESP32-CAM handles: camera activation → capture grayscale VGA → WiFi activation → JPEG compression → HTTP POST to backend  

---

# STM32F401CCU6 Black Pill Camera Unit Pin Connections

> **Version**: Final PCB-Optimized (November 07, 2025)  
> **Key Features**:  
> - SPI1 on PB12–PB15 (left header) for ESP32-CAM.  
> - USART1 on PB6/PB7 (left header) for ESP-01.  
> - Fixed 5V Booster (3.3V → 5V) for power.  
> - 3.3V Regulator from 5V.  
> - HM-10 pre-programmed (VCC + GND only).  
> - Decoupling: Option 2 (100µF electrolytic + 100nF ceramic).  
> - All signals grouped on left header for easy routing.  
> - Single-layer PCB compatible.

---

## Components Recap

| **Component** | **Qty** | **Spec** | **Notes** |
|---------------|--------|----------|-----------|
| Fixed 5V Booster | 1 | 3.3V → 5V, ≥1A | e.g., MT3608 fixed |
| 3.3V Regulator | 1 | AMS1117-3.3 or similar | From 5V booster OUT |
| Electrolytic Capacitor | 1 | 470 µF, 16V, low-ESR | C1 (5V rail) |
| Electrolytic Capacitor | 2 | 100 µF, 16V | C2 (5V rail) + C7 (3.3V rail) |
| Ceramic Capacitor | 2 | 0.1 µF ("104") | C3 (at ESP32-CAM) + C8 (at ESP-01) |
| Resistor | 1 | 10 kΩ, 1/4 W | ESP-01 CH_PD pull-up |
| Battery | 1 | 3.3V Li-ion (e.g., 18650) | ≥2000 mAh |
| ESP32-CAM | 1 | AI-Thinker | |
| STM32F401CCU6 | 1 | Black Pill | |
| ESP-01 | 1 | 1MB flash | AT v2.2.1+ firmware |
| HM-10 | 1 | Pre-programmed | VCC + GND only |
| Jumper Wires / Traces | ~20 | 22–28 AWG / 0.25–1.5 mm | Power: 0.8–1.5 mm; Data: 0.25 mm |

---

## Pin Connection Table

| **From** | **To** | **Wire / Trace** | **Notes** |
|--------|--------|------------------|-----------|
| **3.3V Battery +** | **Fixed 5V Booster IN+** | Red / 1.5 mm trace | |
| **3.3V Battery –** | **COMMON GND** | Black / GND plane | Star point |
| **Fixed 5V Booster IN–** | GND | Black | |
| **Fixed 5V Booster OUT+** | **C1 (+) → C2 (+) → C3 → ESP32-CAM 5V** | Red / 1.5 mm trace | 5.05V |
| **Fixed 5V Booster OUT–** | GND | Black | |
| **C1: 470 µF electrolytic (low-ESR)** | **+** to 5V Booster OUT+ | — | < 3 cm from ESP32-CAM |
| | **–** to GND | — | |
| **C2: 100 µF electrolytic** | **+** to 5V Booster OUT+ | — | < 3 cm from ESP32-CAM |
| | **–** to GND | — | |
| **C3: 100 nF ceramic ("104")** | **Pin 1** to 5V Booster OUT+ | — | **< 1 cm from ESP32-CAM 5V/GND** |
| | **Pin 2** to GND | — | |
| **ESP32-CAM 5V** | C1+/C2+/C3 | Red | |
| **ESP32-CAM GND** | GND | Black | |
| **ESP32-CAM GPIO12 (SCK)** | **STM32 PB13** | Orange / 0.25 mm trace | SPI CLK (unused, PCB compatibility) |
| **ESP32-CAM GPIO13 (MOSI)** | **STM32 PB15** | Purple / 0.25 mm trace | SPI MOSI (unused, PCB compatibility) |
| **ESP32-CAM GPIO14 (MISO)** | **STM32 PB14** | Gray / 0.25 mm trace | SPI MISO (unused, PCB compatibility) |
| **ESP32-CAM GPIO15 (CS)** | **STM32 PB12** | Yellow / 0.25 mm trace | **Trigger signal (100ms HIGH pulse)** |
| **Fixed 5V Booster OUT+** | **3.3V Regulator IN** | Red / 1.5 mm trace | |
| **3.3V Regulator GND** | GND | Black | |
| **3.3V Regulator OUT** | **C7 (+) → STM32 Left 3V3 → STM32 Right 3V3 → ESP-01 VCC → HM-10 VCC** | Red / 0.8 mm trace | Star layout |
| **C7: 100 µF electrolytic** | **+** to 3.3V Regulator OUT | — | |
| | **–** to GND | — | |
| **C8: 0.1 µF ceramic ("104")** | **Pin 1** to ESP-01 VCC | — | **< 1 cm from ESP-01** |
| | **Pin 2** to ESP-01 GND | — | |
| **ESP-01 VCC** | STM32 Right 3V3 | Red | |
| **ESP-01 GND** | GND | Black | |
| **ESP-01 CH_PD** | STM32 Right 3V3 via **10 kΩ** | — | **Must be HIGH** |
| **ESP-01 TX** | **STM32 PB7** | Green / 0.25 mm trace | **USART1_RX** |
| **ESP-01 RX** | **STM32 PB6** | Blue / 0.25 mm trace | **USART1_TX** |
| **HM-10 VCC** | STM32 Right 3V3 | Red | **Pre-programmed** |
| **HM-10 GND** | GND | Black | |
| **STM32 Left 3V3** | C7 (+) | Red | |
| **STM32 Right 3V3** | ESP-01 VCC + HM-10 VCC | Red | |
| **STM32 GND** | GND | Black | |

---

## Power Visual (5V Rail)

```plaintext
3.3V Battery+ ──► [Fixed 5V Booster] ──► [C1: 470µF electrolytic] ──► [C2: 100µF electrolytic] ──► ESP32-CAM 5V
                                             │
                                             └──► [C3: 100 nF ceramic] (at ESP32-CAM pins)
```

## Power Visual (3.3V Rail)

```plaintext
5V Booster OUT+ ──► 3.3V Regulator IN
5V Booster OUT– ──► 3.3V Regulator GND

3.3V Regulator OUT ──► [C7: 100µF electrolytic] ──► STM32 Left 3V3 ──► STM32 Right 3V3 ──► [ESP-01 VCC + HM-10 VCC]
                                 │
                                 └──► [C8: 0.1µF ceramic] (at ESP-01)
```

---

## System Architecture & Responsibilities

### **STM32F401CCU6 (Controller + State Machine)**
- **MQTT Subscription**: Receives occupancy detection from on-slot sensor via ESP-01 (wildcard: `SPARK_C06/isOccupied/*`)
- **Smart Parking Debouncing State Machine**: 
  - IDLE → wait for first occupancy detection (car approaching)
  - STABILIZING → wait for sensor to stabilize (driver aligning car, 2-second window)
  - CONFIRMED → sensor stable, trigger camera once (wide-angle captures all slots)
  - COOLDOWN → 5-second adaptive cooldown (exits early if car leaves)
  - COOLDOWN → IDLE after timeout OR when "empty" received (allows next parking event)
- **Purpose**: 
  - Handles multiple sensor fluctuations during parking alignment
  - Captures only ONE image per parking event (when car is stable)
  - Prevents redundant captures during minor adjustments
  - Fast response: 2s stabilization + 5s adaptive cooldown
  - Multi-slot support: Last car to stabilize triggers capture of all slots in frame
- **Communication**: SPI CS (PB12) to ESP32-CAM GPIO15

### **ESP32-CAM (Image Capture & Upload)**
- **Trigger Detection**: GPIO15 (CS) rising edge interrupt
- **Workflow on trigger**:
  1. Power on camera module (OV2640)
  2. Capture **VGA grayscale (640×480)** image
  3. JPEG compress (quality 75-80) → **~10-15 KB**
  4. Connect to WiFi (keep disconnected when idle to save power)
  5. HTTP POST image to backend (HTTPS supported natively)
  6. Disconnect WiFi
  7. Power down camera
  8. Return to deep sleep
- **Benefits**: 
  - Self-contained image handling
  - **Full VGA resolution** (no downscaling needed)
  - HTTPS support (no ESP-01 limitation)
  - Lower overall system complexity
  - STM32 RAM freed up (no image buffering)
  - ESP32 has sufficient memory (520 KB SRAM) for VGA processing

### **ESP-01 (MQTT Gateway)**
- **Role**: WiFi ↔ MQTT bridge for STM32
- **Responsibilities**:
  - Maintain persistent MQTT connection
  - Subscribe to trigger topic
  - Relay commands to STM32 via UART
  - Publish status messages from STM32
- **Always On**: Keeps MQTT alive for instant triggering

---

## Image Processing Strategy (ESP32-CAM)

- **Resolution**: VGA (640×480) grayscale
- **Algorithm**: JPEG compression (quality 75–80)
- **File Size**: ~10–15 KB (grayscale VGA with JPEG compression)
- **Rationale**: Full VGA resolution for optimal YOLOv2 detection accuracy
- **Implementation**: ESP32 built-in JPEG encoder
- **Memory**: ESP32 has 520 KB SRAM (sufficient for VGA framebuffer + JPEG encoder)
- **Encoding Time**: ~200-300 ms on ESP32 @ 240 MHz
- **HTTP Upload**: Direct HTTPS POST to Hugging Face backend
- **Power Management**: Camera/WiFi disabled between triggers to conserve battery

---

## Smart Parking Debouncing State Machine (STM32)

**Real-World Scenario**: Driver approaches spot → sensor detects → driver reverses to align → sensor clears → driver moves forward → sensor detects again → repeat 2-3 times → car finally parked.

**Goal**: Capture ONE image only when car is stable and parked, not during alignment maneuvers.

```
┌─────────────────────────────────────────────────┐
│                    IDLE                         │
│  - No car detected (spot empty)                 │
│  - PB12 = LOW                                    │
│  - Waiting for first occupancy detection        │
└──────────────┬──────────────────────────────────┘
               │ MQTT: "1" (occupied) received
               ▼
┌─────────────────────────────────────────────────┐
│                STABILIZING                       │
│  - Car detected (driver aligning)               │
│  - PB12 = LOW (camera NOT triggered yet)        │
│  - Timer: 2 seconds (faster multi-slot)         │
│  - Any new "1" resets timer (multi-car support) │
│  - If "0" (empty) received → back to IDLE       │
└──────────────┬──────────────────────────────────┘
               │ 2s elapsed without state change
               ▼
┌─────────────────────────────────────────────────┐
│                 CONFIRMED                        │
│  - Car stable → trigger camera                  │
│  - Assert PB12 = HIGH for 100ms                 │
│  - ESP32-CAM captures all slots (wide-angle)    │
└──────────────┬──────────────────────────────────┘
               │ After 100ms
               ▼
┌─────────────────────────────────────────────────┐
│                 COOLDOWN                         │
│  - PB12 = LOW                                    │
│  - Ignores "1" messages for 5s minimum          │
│  - "0" (empty) → IMMEDIATE exit to IDLE         │
│  - Duration: 5s adaptive (exits early)          │
│  - Adaptive behavior allows fast queue:         │
│    • Car leaves after 2s → only 2s wait         │
│    • Next car can trigger immediately           │
└──────────────┬──────────────────────────────────┘
               │ After 5s timeout OR "0" received
               └──────► Back to IDLE (ready for next car)
```

**State Transitions**:
- **IDLE → STABILIZING**: First "occupied" detection
- **STABILIZING → STABILIZING**: Additional "occupied" resets 3s timer (car still moving)
- **STABILIZING → IDLE**: "empty" received (car left before parking)
- **STABILIZING → CONFIRMED**: 3s elapsed with stable "occupied" state
- **CONFIRMED → COOLDOWN**: Camera triggered, enter 30s ignore period
- **COOLDOWN → IDLE**: Timeout, ready for next parking event

**Example Timeline** (Camera monitoring slots A01-A05):
```
00:00 - IDLE (all spots empty)
00:05 - MQTT: "SPARK_C06/isOccupied/A01" payload "1" → STABILIZING (start 3s timer)
00:07 - MQTT: "SPARK_C06/isOccupied/A01" payload "1" → STABILIZING (reset timer, car moving)
00:08 - MQTT: "SPARK_C06/isOccupied/A01" payload "0" → IDLE (car reversed out)
00:10 - MQTT: "SPARK_C06/isOccupied/A03" payload "1" → STABILIZING (different slot detected)
00:12 - MQTT: "SPARK_C06/isOccupied/A03" payload "1" → STABILIZING (reset timer)
00:15 - (3s elapsed) → CONFIRMED (trigger camera, captures A03)
00:15.1 - COOLDOWN (30s ignore period - ignores A01, A02, A04, A05 during this time)
00:45 - IDLE (ready for next parking event in ANY monitored slot)
```

**System Architecture**:
- **On-Slot Sensors**: Each publishes to `SPARK_C06/isOccupied/{slotID}` with `"1"` or `"0"`
  - Sensor A01 → `SPARK_C06/isOccupied/A01`
  - Sensor A03 → `SPARK_C06/isOccupied/A03`
  - Sensor B12 → `SPARK_C06/isOccupied/B12`
- **Camera Unit**: Subscribes to `SPARK_C06/isOccupied/*` (receives ALL slot updates)
  - Triggers on ANY slot showing parking activity
  - Captures wide-angle image covering all monitored slots
  - ESP32-CAM can use CV to identify which specific slot changed
- **Backend**: Subscribes to `SPARK_C06/isOccupied/#` 
  - Parses topic to extract slot ID
  - Updates database: `UPDATE parking_slots SET occupied = ? WHERE slot_id = ?`

---

## Communication Protocol

### **STM32 → ESP32-CAM (Trigger Signal)**
- **Interface**: SPI pins (PB12-PB15) - only CS (PB12/GPIO15) used as trigger
- **Trigger Pin**: STM32 PB12 → ESP32-CAM GPIO15
- **Protocol**: 100ms HIGH pulse on CS line
- **ESP32-CAM**: Configured with rising edge interrupt on GPIO15
- **SPI Data Lines**: Present on PCB (SCK/MOSI/MISO) but unused in current implementation
- **PCB Compatibility**: All SPI traces maintained for potential future bidirectional communication

### **ESP-01 ↔ STM32 (UART + MQTT)**
- **Baud Rate**: 115200
- **Protocol**: AT commands + raw MQTT packets
- **TX (STM32 PB6)** → ESP-01 RX
- **RX (STM32 PB7)** ← ESP-01 TX
- **MQTT Topics**:
  - Subscribe: `SPARK_C06/isOccupied/*` (wildcard - receives ALL slot occupancy updates)
    - Matches: `SPARK_C06/isOccupied/A01`, `SPARK_C06/isOccupied/B12`, etc.
  - Publish: `SPARK_C06/camera/status` (publishes camera system status)
  - **Payloads**: 
    - `"1"` - Slot occupied (car detected)
    - `"0"` - Slot empty (no car)
  - **Benefits**: 
    - One camera monitors multiple parking slots in its view
    - Simple binary format for backend database updates
    - No need to reconfigure camera firmware when adding/removing slots
    - On-slot sensors publish to specific topics (A01, B05, etc.)
    - Camera triggers on ANY occupancy change in its monitoring zone

### **ESP32-CAM → Backend (HTTP)**
- **Protocol**: HTTPS POST
- **Endpoint**: `https://danishritonga-spark-backend.hf.space/upload`
- **Content-Type**: `multipart/form-data` or `application/octet-stream`
- **Payload**: JPEG image (~10-12 KB)

---

## Notes

- **PCB Routing**: Full SPI + UART connections on left header → easy single-layer.  
- **SPI Usage**: Only CS (PB12) used as trigger; SCK/MOSI/MISO present but unused.
- **Trace Widths**: 5V: 1.5 mm; 3.3V: 0.8 mm; Data: 0.25 mm.  
- **Cap Placement**: C3/C8 <1 cm from loads.  
- **Optional Zener**: If booster spikes, add 5.1V Zener on 5V rail.  
- **Test**: Measure 5V/3.3V under load → stable ±0.05V.
- **Debouncing**: 5-second cooldown prevents rapid repeated triggers.
- **Power Efficiency**: ESP32-CAM in deep sleep between triggers, WiFi only active during upload.
- **Image Resolution**: VGA (640×480) supported by ESP32's 520 KB SRAM - no memory constraints.  
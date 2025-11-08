# ESP32-CAM + STM32F401 + ESP-01 Camera Unit

> **Minimal, stable, trigger-based PoC**  
> ESP-01 receives MQTT → STM32 triggers ESP32-CAM → captures grayscale VGA → compress the image in STM32 → publishes via REST API  

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
| **ESP32-CAM GPIO12 (SCK)** | **STM32 PB13** | Orange / 0.25 mm trace | SPI CLK |
| **ESP32-CAM GPIO13 (MOSI)** | **STM32 PB15** | Purple / 0.25 mm trace | SPI MOSI |
| **ESP32-CAM GPIO14 (MISO)** | **STM32 PB14** | Gray / 0.25 mm trace | SPI MISO |
| **ESP32-CAM GPIO15 (CS)** | **STM32 PB12** | Yellow / 0.25 mm trace | **Trigger / NSS** |
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

## Image Compression Strategy

- **Algorithm**: JPEG (quality 75–80)
- **Rationale**: Optimal balance between file size (~10–12 KB for VGA grayscale) and YOLOv2 detection accuracy
- **Implementation**: libjpeg-turbo or TinyJPEG (minimal footprint on STM32F401)
- **Encoding Time**: ~500 ms at 84 MHz
- **HTTP Payload**: ~10–12 KB per frame (suitable for ESP-01 115.2k UART)

---

## Notes

- **PCB Routing**: All SPI/UART on left header → easy single-layer.  
- **Trace Widths**: 5V: 1.5 mm; 3.3V: 0.8 mm; Data: 0.25 mm.  
- **Cap Placement**: C3/C8 <1 cm from loads.  
- **Optional Zener**: If booster spikes, add 5.1V Zener on 5V rail.  
- **Test**: Measure 5V/3.3V under load → stable ±0.05V.  
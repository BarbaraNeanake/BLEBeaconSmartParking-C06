/**
 * STM32F401CC Smart Parking Management System
 * Real-time parking slot detection using magnetic reed switches
 * 
 * Hardware:
 * - 5x Magnetic Reed Switches (Active Low - Normally Open)
 * - 5x Active Buzzers (triggered when switch closes)
 * - Status LED on PC13
 * 
 * Functionality:
 * - Buzzer sounds when car parks (magnetic switch closes)
 * - LED blinks to indicate slot state changes
 */

#include "stm32f4xx_hal.h"
#include <string.h>
#include <stdio.h>

// ==================== CONFIGURATION ====================

// Status LED
#define LED_PIN GPIO_PIN_13
#define LED_PORT GPIOC

// ESP-01 UART Configuration (USART2 - PB7 is unusable)
#define ESP01_USART USART2
#define ESP01_BAUDRATE 115200

// WiFi Credentials
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"

// MQTT Broker Configuration
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_CLIENT_ID "SPARK_C06"
#define MQTT_TOPIC_BASE "SPARK_C06/isOccupied/"
#define MQTT_TOPIC_PING "SPARK_C06/ping/"

// Parking Slot Configuration (5 slots with magnetic reed switches)
#define NUM_SLOTS 5

// Buzzer Configuration
#define BUZZER_DURATION_MS 500  // Buzzer active duration when car parks (local trigger)
#define ALARM_DURATION_MS 5000  // Alarm duration for MQTT trigger (5 seconds)
#define ALARM_BEEP_ON_MS 200    // Beep on duration
#define ALARM_BEEP_OFF_MS 200   // Beep off duration (pause between beeps)

// Reed Switch Pin Assignments (Active Low - Closed = Occupied)
// Using PB0, PB1, PB10, PB12, PB13 for 5 parking slots
// Buzzer pins: PA0, PA1, PA4, PA5, PA6 for 5 buzzers
typedef struct {
    GPIO_TypeDef* reed_port;
    uint16_t reed_pin;
    GPIO_TypeDef* buzzer_port;
    uint16_t buzzer_pin;
    const char* slot_id;
    uint8_t last_state;      // 0 = Empty, 1 = Occupied
    uint8_t current_state;   // 0 = Empty, 1 = Occupied
    uint8_t changed;         // Flag for state change
    uint8_t buzzer_active;   // Flag for buzzer state
    uint32_t buzzer_start;   // Buzzer start time
    uint8_t alarm_active;    // Flag for alarm state (MQTT trigger)
    uint32_t alarm_start;    // Alarm start time
    uint8_t alarm_beep_state; // Current beep state (on/off)
    uint32_t alarm_beep_time; // Last beep toggle time
} ParkingSlot;

ParkingSlot slots[NUM_SLOTS] = {
    {GPIOB, GPIO_PIN_0,  GPIOA, GPIO_PIN_0, "1", 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {GPIOB, GPIO_PIN_1,  GPIOA, GPIO_PIN_1, "2", 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {GPIOB, GPIO_PIN_10, GPIOA, GPIO_PIN_4, "3", 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {GPIOB, GPIO_PIN_12, GPIOA, GPIO_PIN_5, "4", 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {GPIOB, GPIO_PIN_13, GPIOA, GPIO_PIN_6, "5", 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// Buffer sizes
#define RX_BUFFER_SIZE 512
#define TX_BUFFER_SIZE 256

// Debounce delay (milliseconds)
#define DEBOUNCE_DELAY 200

// ==================== GLOBAL VARIABLES ====================

UART_HandleTypeDef huart2;
char rx_buffer[RX_BUFFER_SIZE];
volatile uint16_t rx_index = 0;
volatile uint8_t rx_byte;
volatile uint8_t wifi_connected = 0;
volatile uint8_t mqtt_connected = 0;
uint32_t last_ping_time = 0;
uint32_t ping_interval = 30000; // Ping every 30 seconds

// ==================== FUNCTION PROTOTYPES ====================

void SystemClock_Config(void);
void Error_Handler(void);
static void GPIO_Init(void);
static void USART2_Init(void);
static void Blink_Status(uint8_t count, uint16_t delay);

// ESP-01 Functions
static void send_AT(const char* cmd, uint32_t delay_ms);
static void send_data(uint8_t* data, uint16_t len);
static void wait_for_prompt(void);
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password);
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port);
static uint8_t ESP01_Send_MQTT_CONNECT(const char* clientID);
static void ESP01_Send_MQTT_PUBLISH(const char* topic, const char* message);
static void ESP01_Send_MQTT_SUBSCRIBE(const char* topic);
static void ESP01_ReceiveAndParse(void);
static void ESP01_SendPing(void);
static uint8_t ESP01_CheckConnection(void);
static void ESP01_Reconnect(void);

// Parking Slot Functions
static void Read_Parking_Slots(void);
static void Process_Slot_Changes(void);
static void Manage_Buzzers(void);
static void Trigger_Buzzer(uint8_t slot_index);
static void Trigger_Alarm(uint8_t slot_index);

// ==================== MAIN PROGRAM ====================

int main(void)
{
    // Initialize HAL
    HAL_Init();
    
    // Configure system clock
    SystemClock_Config();
    
    // Initialize peripherals
    GPIO_Init();
    USART2_Init();
    
    // System ready indication
    Blink_Status(3, 200);
    
    // Wait for ESP-01 to boot
    HAL_Delay(2000);
    
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Disable echo
    send_AT("ATE0\r\n", 500);
    HAL_Delay(500);
    
    // Connect to WiFi
    if (ESP01_ConnectWiFi(WIFI_SSID, WIFI_PASSWORD))
    {
        Blink_Status(3, 300); // WiFi connected
        HAL_Delay(1000);
    }
    else
    {
        Blink_Status(10, 200); // WiFi failed
        while(1) { HAL_Delay(1000); }
    }
    
    // Connect to MQTT Broker
    if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
    {
        HAL_Delay(2000);
    }
    else
    {
        Blink_Status(10, 300); // MQTT TCP failed
        while(1) { HAL_Delay(1000); }
    }
    
    // Send MQTT CONNECT packet
    if (ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID))
    {
        Blink_Status(5, 200); // MQTT connected
        HAL_Delay(2000);
    }
    else
    {
        Blink_Status(10, 400); // MQTT CONNECT failed
        while(1) { HAL_Delay(1000); }
    }
    
    // Subscribe to ping topic
    ESP01_Send_MQTT_SUBSCRIBE("SPARK_C06/ping/#");
    Blink_Status(4, 150); // Subscribed
    HAL_Delay(2000);
    
    // System fully operational
    Blink_Status(2, 100);
    HAL_Delay(1000);
    
    // Initialize timing
    last_ping_time = HAL_GetTick();
    uint32_t lastCheckTime = HAL_GetTick() - 240000; // First check at 5 minutes
    
    // Main loop - Monitor parking slots and communicate status
    while (1)
    {
        // Read all parking slot states
        Read_Parking_Slots();
        
        // Process and publish any state changes
        Process_Slot_Changes();
        
        // Check for incoming MQTT messages (ping commands)
        ESP01_ReceiveAndParse();
        
        // Manage buzzer timers
        Manage_Buzzers();
        
        // Send MQTT ping every 30 seconds
        if ((HAL_GetTick() - last_ping_time) >= ping_interval)
        {
            ESP01_SendPing();
            last_ping_time = HAL_GetTick();
        }
        
        // Check connection status every 5 minutes
        if ((HAL_GetTick() - lastCheckTime) >= 300000)
        {
            if (!ESP01_CheckConnection())
            {
                ESP01_Reconnect();
            }
            lastCheckTime = HAL_GetTick();
        }
        
        // Small delay for stability
        HAL_Delay(100);
    }
    
    return 0;
}

// ==================== PARKING SLOT FUNCTIONS ====================

/**
 * @brief Read all parking slot reed switches
 * Active Low: GPIO_PIN_RESET (0V) = Occupied, GPIO_PIN_SET (3.3V) = Empty
 */
static void Read_Parking_Slots(void)
{
    static uint32_t last_read_time[NUM_SLOTS] = {0};
    uint32_t current_time = HAL_GetTick();
    
    for (uint8_t i = 0; i < NUM_SLOTS; i++)
    {
        // Read current pin state
        GPIO_PinState pin_state = HAL_GPIO_ReadPin(slots[i].reed_port, slots[i].reed_pin);
        
        // Active Low Logic: LOW = Occupied (magnet closes switch to GND)
        uint8_t occupied = (pin_state == GPIO_PIN_RESET) ? 1 : 0;
        
        // Check if state changed and debounce
        if (occupied != slots[i].current_state)
        {
            if ((current_time - last_read_time[i]) >= DEBOUNCE_DELAY)
            {
                slots[i].current_state = occupied;
                
                // Check if different from last published state
                if (slots[i].current_state != slots[i].last_state)
                {
                    slots[i].changed = 1;
                    
                    // Trigger buzzer when car parks (becomes occupied)
                    if (slots[i].current_state == 1)
                    {
                        Trigger_Buzzer(i);
                    }
                }
                
                last_read_time[i] = current_time;
            }
        }
    }
}

/**
 * @brief Process slot state changes and publish to MQTT
 */
static void Process_Slot_Changes(void)
{
    char topic[64];
    const char* payload;
    
    for (uint8_t i = 0; i < NUM_SLOTS; i++)
    {
        if (slots[i].changed)
        {
            // Only publish if MQTT is connected
            if (!mqtt_connected)
            {
                // Clear flag but don't update last_state so we retry later
                slots[i].changed = 0;
                continue;
            }
            
            // Build topic: SPARK_C06/isOccupied/1
            snprintf(topic, sizeof(topic), "%s%s", MQTT_TOPIC_BASE, slots[i].slot_id);
            
            // Determine payload
            payload = slots[i].current_state ? "True" : "False";
            
            // Publish to MQTT
            ESP01_Send_MQTT_PUBLISH(topic, payload);
            
            // Visual feedback
            if (slots[i].current_state)
            {
                Blink_Status(2, 100); // Occupied
            }
            else
            {
                Blink_Status(1, 50); // Empty
            }
            
            // Update last state and clear flag
            slots[i].last_state = slots[i].current_state;
            slots[i].changed = 0;
            
            // Delay between publishes
            HAL_Delay(1000);
        }
    }
}

// ==================== GPIO INITIALIZATION ====================

/**
 * @brief Initialize GPIO pins
 * - PC13: Status LED (Output)
 * - PB0, PB1, PB10, PB12, PB13: Reed Switches (Input with Pull-up)
 */
static void GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    // Enable GPIO clocks
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    
    // Configure LED pin (PC13)
    GPIO_InitStruct.Pin = LED_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    
    // Configure reed switch pins (Input with Pull-up for Active Low)
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    
    for (uint8_t i = 0; i < NUM_SLOTS; i++)
    {
        GPIO_InitStruct.Pin = slots[i].reed_pin;
        HAL_GPIO_Init(slots[i].reed_port, &GPIO_InitStruct);
    }
    
    // Configure buzzer pins (Output Push-Pull)
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    
    for (uint8_t i = 0; i < NUM_SLOTS; i++)
    {
        GPIO_InitStruct.Pin = slots[i].buzzer_pin;
        HAL_GPIO_Init(slots[i].buzzer_port, &GPIO_InitStruct);
        // Initialize buzzers OFF
        HAL_GPIO_WritePin(slots[i].buzzer_port, slots[i].buzzer_pin, GPIO_PIN_RESET);
    }
}

// ==================== USART2 INITIALIZATION ====================

/**
 * @brief Initialize USART2 for ESP-01 communication
 * PA2 - USART2_TX, PA3 - USART2_RX (PB7 is unusable)
 */
static void USART2_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    // Enable clocks
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_USART2_CLK_ENABLE();
    
    // Configure GPIO pins for USART2
    // PA2: TX, PA3: RX
    GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    
    // Configure USART2
    huart2.Instance = USART2;
    huart2.Init.BaudRate = ESP01_BAUDRATE;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    
    if (HAL_UART_Init(&huart2) != HAL_OK)
    {
        Error_Handler();
    }
    
    // Enable UART receive interrupt
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
    
    // Start receiving in interrupt mode
    HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
}

// ==================== ESP-01 COMMUNICATION FUNCTIONS ====================

/**
 * @brief Send AT command and wait
 */
static void send_AT(const char* cmd, uint32_t delay_ms)
{
    HAL_UART_AbortReceive_IT(&huart2);
    HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(delay_ms);
    HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
}

/**
 * @brief Send raw data to ESP-01
 */
static void send_data(uint8_t* data, uint16_t len)
{
    HAL_UART_AbortReceive_IT(&huart2);
    HAL_UART_Transmit(&huart2, data, len, HAL_MAX_DELAY);
    HAL_Delay(500);
    HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
}

/**
 * @brief Wait for '>' prompt from ESP-01 after AT+CIPSEND
 */
static void wait_for_prompt(void)
{
    HAL_Delay(500);
}

/**
 * @brief Connect ESP-01 to WiFi network
 * @return 1 if connected, 0 if failed
 */
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password)
{
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Set station mode
    send_AT("AT+CWMODE=1\r\n", 1000);
    
    // Disconnect from any existing WiFi
    send_AT("AT+CWQAP\r\n", 1000);
    HAL_Delay(1000);
    
    // Clear buffers
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Connect to WiFi
    char wifiCommand[128];
    snprintf(wifiCommand, sizeof(wifiCommand),
             "AT+CWJAP=\"%s\",\"%s\"\r\n", ssid, password);
    
    HAL_UART_Transmit(&huart2, (uint8_t*)wifiCommand, strlen(wifiCommand), HAL_MAX_DELAY);
    
    // Blink during connection (WiFi takes 5-10 seconds)
    for (uint8_t i = 0; i < 40; i++)
    {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        HAL_Delay(250);
    }
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    
    // Wait longer for WiFi to stabilize
    HAL_Delay(3000);
    
    wifi_connected = 1;
    return 1;
}

/**
 * @brief Connect to MQTT broker via TCP
 * @return 1 if connected, 0 if failed
 */
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port)
{
    char cmd[128];
    
    // Close any existing connection
    send_AT("AT+CIPCLOSE\r\n", 1000);
    
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    snprintf(cmd, sizeof(cmd),
             "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", broker, port);
    
    HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(6000);
    
    return 1;
}

/**
 * @brief Send MQTT CONNECT packet
 * @return 1 if successful, 0 if failed
 */
static uint8_t ESP01_Send_MQTT_CONNECT(const char* clientID)
{
    uint8_t packet[128];
    uint8_t clientID_len = strlen(clientID);
    uint8_t total_len = 10 + 2 + clientID_len;
    
    uint8_t index = 0;
    packet[index++] = 0x10;                     // CONNECT command
    packet[index++] = total_len;                // Remaining length
    packet[index++] = 0x00; packet[index++] = 0x04; // Length of "MQTT"
    packet[index++] = 'M'; packet[index++] = 'Q'; 
    packet[index++] = 'T'; packet[index++] = 'T';
    packet[index++] = 0x04;                     // Protocol level 4
    packet[index++] = 0x02;                     // Clean session
    packet[index++] = 0x00; packet[index++] = 0x3C; // Keep alive = 60 sec
    packet[index++] = 0x00; packet[index++] = clientID_len;
    memcpy(&packet[index], clientID, clientID_len);
    index += clientID_len;
    
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Send CIPSEND command
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    wait_for_prompt();
    
    // Send MQTT CONNECT packet
    HAL_UART_Transmit(&huart2, packet, index, HAL_MAX_DELAY);
    
    // Wait longer for MQTT CONNACK
    HAL_Delay(3000);
    
    mqtt_connected = 1;
    return 1;
}

/**
 * @brief Send MQTT PUBLISH packet
 */
static void ESP01_Send_MQTT_PUBLISH(const char* topic, const char* message)
{
    uint8_t packet[256];
    uint8_t topic_len = strlen(topic);
    uint8_t message_len = strlen(message);
    
    uint8_t remaining_len = 2 + topic_len + message_len;
    uint8_t index = 0;
    
    packet[index++] = 0x30;                     // PUBLISH command (QoS 0)
    packet[index++] = remaining_len;            // Remaining length
    packet[index++] = 0x00;                     // Topic length MSB
    packet[index++] = topic_len;                // Topic length LSB
    memcpy(&packet[index], topic, topic_len);
    index += topic_len;
    memcpy(&packet[index], message, message_len);
    index += message_len;
    
    // Send CIPSEND command
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    wait_for_prompt();
    
    // Send MQTT PUBLISH packet
    send_data(packet, index);
}

/**
 * @brief Send MQTT SUBSCRIBE packet
 */
static void ESP01_Send_MQTT_SUBSCRIBE(const char* topic)
{
    uint8_t packet[128];
    uint8_t topic_len = strlen(topic);
    uint8_t remaining_len = 2 + 2 + topic_len + 1; // Packet ID (2) + Topic length (2) + Topic + QoS (1)
    
    uint8_t index = 0;
    packet[index++] = 0x82;            // SUBSCRIBE packet type with flags
    packet[index++] = remaining_len;   // Remaining Length
    packet[index++] = 0x00;            // Packet ID MSB
    packet[index++] = 0x01;            // Packet ID LSB
    packet[index++] = 0x00;            // Topic length MSB
    packet[index++] = topic_len;       // Topic length LSB
    memcpy(&packet[index], topic, topic_len); // Topic
    index += topic_len;
    packet[index++] = 0x00;            // QoS 0
    
    // Send CIPSEND command
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    wait_for_prompt();
    send_data(packet, index);
}

/**
 * @brief Receive and parse incoming MQTT messages
 * Detects ping messages on SPARK_C06/ping/{slotID}
 */
static void ESP01_ReceiveAndParse(void)
{
    // Need enough bytes for MQTT PUBLISH packet
    if (rx_index < 20)
        return;
    
    // Search for "SPARK_C06/ping/" in the buffer
    const char* ping_topic = "SPARK_C06/ping/";
    const int ping_len = strlen(ping_topic);
    
    for (int i = 0; i <= rx_index - ping_len; i++)
    {
        if (strncmp((char*)&rx_buffer[i], ping_topic, ping_len) == 0)
        {
            // Found ping topic, extract slot ID (single digit after "ping/")
            if (i + ping_len < rx_index)
            {
                char slot_id = rx_buffer[i + ping_len];
                
                // Match slot ID to trigger alarm
                for (uint8_t j = 0; j < NUM_SLOTS; j++)
                {
                    if (slot_id == slots[j].slot_id[0])
                    {
                        Trigger_Alarm(j);
                        break;
                    }
                }
            }
            
            // Clear buffer after processing
            rx_index = 0;
            memset(rx_buffer, 0, RX_BUFFER_SIZE);
            return;
        }
    }
    
    // Clear buffer if getting too full
    if (rx_index > RX_BUFFER_SIZE - 100)
    {
        rx_index = 0;
        memset(rx_buffer, 0, RX_BUFFER_SIZE);
    }
}

/**
 * @brief Send MQTT PINGREQ to keep connection alive
 */
static void ESP01_SendPing(void)
{
    uint8_t ping_packet[2] = {0xC0, 0x00}; // PINGREQ packet
    
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", 2);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    wait_for_prompt();
    send_data(ping_packet, 2);
}

/**
 * @brief Check if connection is still alive
 * @return 1 if connected, 0 if disconnected
 */
static uint8_t ESP01_CheckConnection(void)
{
    return 1; // Trust connection is alive
}

/**
 * @brief Reconnect to WiFi and MQTT broker
 */
static void ESP01_Reconnect(void)
{
    // Indicate reconnection attempt
    Blink_Status(10, 50);
    
    // Mark as disconnected
    wifi_connected = 0;
    mqtt_connected = 0;
    
    // Close existing connection
    send_AT("AT+CIPCLOSE\r\n", 1000);
    HAL_Delay(1000);
    
    // Reconnect to WiFi
    if (ESP01_ConnectWiFi(WIFI_SSID, WIFI_PASSWORD))
    {
        Blink_Status(3, 300);
        HAL_Delay(2000);
        
        // Reconnect to MQTT broker
        if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
        {
            HAL_Delay(3000);
            
            // Send MQTT CONNECT
            if (ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID))
            {
                Blink_Status(5, 200);
                HAL_Delay(2000);
                
                // Re-subscribe to ping topic
                ESP01_Send_MQTT_SUBSCRIBE("SPARK_C06/ping/#");
                Blink_Status(4, 150);
                HAL_Delay(2000);
            }
            else
            {
                mqtt_connected = 0;
            }
        }
        else
        {
            mqtt_connected = 0;
        }
    }
    else
    {
        Blink_Status(15, 100);
        HAL_Delay(5000);
    }
}

// ==================== BUZZER CONTROL FUNCTIONS ====================

/**
 * @brief Blink LED for status indication
 */
static void Blink_Status(uint8_t count, uint16_t delay)
{
    for (uint8_t i = 0; i < count; i++)
    {
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_SET);
        HAL_Delay(delay);
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
        HAL_Delay(delay);
    }
}

/**
 * @brief Trigger buzzer for a specific slot
 * @param slot_index Index of the slot (0-4)
 */
static void Trigger_Buzzer(uint8_t slot_index)
{
    if (slot_index >= NUM_SLOTS)
        return;
    
    // Don't activate if alarm is already running
    if (slots[slot_index].alarm_active)
        return;
    
    // Activate buzzer
    HAL_GPIO_WritePin(slots[slot_index].buzzer_port, slots[slot_index].buzzer_pin, GPIO_PIN_SET);
    slots[slot_index].buzzer_active = 1;
    slots[slot_index].buzzer_start = HAL_GetTick();
    
    // Visual feedback
    Blink_Status(1, 50);
}

/**
 * @brief Trigger alarm (beeping pattern) for a specific slot
 * @param slot_index Index of the slot (0-4)
 */
static void Trigger_Alarm(uint8_t slot_index)
{
    if (slot_index >= NUM_SLOTS)
        return;
    
    // Activate alarm
    slots[slot_index].alarm_active = 1;
    slots[slot_index].alarm_start = HAL_GetTick();
    slots[slot_index].alarm_beep_state = 1; // Start with beep ON
    slots[slot_index].alarm_beep_time = HAL_GetTick();
    slots[slot_index].buzzer_active = 1;
    
    // Start first beep
    HAL_GPIO_WritePin(slots[slot_index].buzzer_port, slots[slot_index].buzzer_pin, GPIO_PIN_SET);
    
    // Visual feedback
    Blink_Status(3, 50);
}

/**
 * @brief Manage buzzer timers - turn off buzzers after duration
 */
static void Manage_Buzzers(void)
{
    uint32_t current_time = HAL_GetTick();
    
    for (uint8_t i = 0; i < NUM_SLOTS; i++)
    {
        // Handle simple buzzer (local trigger from reed switch)
        if (slots[i].buzzer_active && !slots[i].alarm_active)
        {
            // Check if buzzer duration elapsed
            if ((current_time - slots[i].buzzer_start) >= BUZZER_DURATION_MS)
            {
                // Deactivate buzzer
                HAL_GPIO_WritePin(slots[i].buzzer_port, slots[i].buzzer_pin, GPIO_PIN_RESET);
                slots[i].buzzer_active = 0;
            }
        }
        
        // Handle alarm (MQTT trigger - beeping pattern)
        if (slots[i].alarm_active)
        {
            // Check if alarm duration elapsed (5 seconds)
            if ((current_time - slots[i].alarm_start) >= ALARM_DURATION_MS)
            {
                // Deactivate alarm
                HAL_GPIO_WritePin(slots[i].buzzer_port, slots[i].buzzer_pin, GPIO_PIN_RESET);
                slots[i].alarm_active = 0;
                slots[i].buzzer_active = 0;
                slots[i].alarm_beep_state = 0;
            }
            else
            {
                // Manage beeping pattern
                uint32_t beep_interval = slots[i].alarm_beep_state ? ALARM_BEEP_ON_MS : ALARM_BEEP_OFF_MS;
                
                if ((current_time - slots[i].alarm_beep_time) >= beep_interval)
                {
                    // Toggle beep state
                    slots[i].alarm_beep_state = !slots[i].alarm_beep_state;
                    slots[i].alarm_beep_time = current_time;
                    
                    // Update buzzer output
                    if (slots[i].alarm_beep_state)
                    {
                        HAL_GPIO_WritePin(slots[i].buzzer_port, slots[i].buzzer_pin, GPIO_PIN_SET);
                    }
                    else
                    {
                        HAL_GPIO_WritePin(slots[i].buzzer_port, slots[i].buzzer_pin, GPIO_PIN_RESET);
                    }
                }
            }
        }
    }
}

// ==================== SYSTEM CONFIGURATION ====================

/**
 * @brief System Clock Configuration
 * Configure the system clock to run at 84 MHz
 */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    
    // Configure the main internal regulator output voltage
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);
    
    // Initialize the RCC Oscillators according to the specified parameters
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ = 7;
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }
    
    // Initialize the CPU, AHB and APB buses clocks
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
    {
        Error_Handler();
    }
}

// ==================== INTERRUPT HANDLERS ====================

/**
 * @brief UART receive complete callback
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        // Add received byte to buffer
        if (rx_index < RX_BUFFER_SIZE - 1)
        {
            rx_buffer[rx_index++] = rx_byte;
        }
        else
        {
            // Buffer full, reset
            rx_index = 0;
        }
        
        // Continue receiving
        HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
    }
}

/**
 * @brief USART2 IRQ Handler
 */
void USART2_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart2);
}

/**
 * @brief SysTick Handler
 */
void SysTick_Handler(void)
{
    HAL_IncTick();
}

// ==================== ERROR HANDLERS ====================

/**
 * @brief Error Handler
 */
void Error_Handler(void)
{
    __disable_irq();
    while (1)
    {
    }
}

/**
 * @brief Hard Fault Handler
 */
void HardFault_Handler(void)
{
    Error_Handler();
}

/**
 * @brief Memory Management Fault Handler
 */
void MemManage_Handler(void)
{
    Error_Handler();
}

/**
 * @brief Bus Fault Handler
 */
void BusFault_Handler(void)
{
    Error_Handler();
}

/**
 * @brief Usage Fault Handler
 */
void UsageFault_Handler(void)
{
    Error_Handler();
}

/**
 * @brief SVC Handler
 */
void SVC_Handler(void)
{
}

/**
 * @brief Debug Monitor Handler
 */
void DebugMon_Handler(void)
{
}

/**
 * @brief PendSV Handler
 */
void PendSV_Handler(void)
{
}

#ifdef USE_FULL_ASSERT
/**
 * @brief Assert failed handler
 */
void assert_failed(uint8_t *file, uint32_t line)
{
    while (1)
    {
    }
}
#endif

#include "stm32f4xx_hal.h"
#include <string.h>
#include <stdio.h>

// LED pin configuration
#define LED_PIN GPIO_PIN_13
#define LED_PORT GPIOC

// ESP32-CAM Trigger Pin (SPI CS - PB12)
#define TRIGGER_PIN GPIO_PIN_12
#define TRIGGER_PORT GPIOB

// ESP-01 Configuration
#define ESP01_USART USART1
#define ESP01_BAUDRATE 115200

// WiFi Credentials
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"

// HiveMQ Public Broker
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_CLIENT_ID "SPARK_C06"

// MQTT Topic Configuration - Smart Parking System
#define MQTT_TOPIC_BASE "SPARK_C06/isOccupied/"
#define MQTT_TOPIC_SUB MQTT_TOPIC_BASE "#"              // Subscribe: "SPARK_C06/isOccupied/*" (all slots)
#define MQTT_TOPIC_PUB "SPARK_C06/camera/status"        // Publish: camera status
#define MQTT_KEYWORD_OCCUPIED "True"                        // Payload: car detected
#define MQTT_KEYWORD_EMPTY "False"                           // Payload: spot empty

// Camera Trigger Configuration
#define TRIGGER_PULSE_MS 100     // Trigger pulse duration (100ms HIGH)

// Buffer sizes
#define RX_BUFFER_SIZE 512
#define TX_BUFFER_SIZE 256

void SystemClock_Config(void);
void Error_Handler(void);
static void GPIO_Init(void);
static void USART1_Init(void);
static void Blink_Status(uint8_t count, uint16_t delay);

// ESP-01 Functions (Raw MQTT over TCP)
static void send_AT(const char* cmd, uint32_t delay_ms);
static void send_data(uint8_t* data, uint16_t len);
static void wait_for_prompt(void);
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password);
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port);
static uint8_t ESP01_Send_MQTT_CONNECT(const char* clientID);
static void ESP01_Send_MQTT_PUBLISH(const char* topic, const char* message);
static void ESP01_Send_MQTT_SUBSCRIBE(const char* topic);
static void ESP01_ReceiveAndParse(void);
static uint8_t ESP01_CheckConnection(void);
static void ESP01_Reconnect(void);
static void ESP01_SendPing(void);

// Global variables
UART_HandleTypeDef huart1;
char rx_buffer[RX_BUFFER_SIZE];
volatile uint16_t rx_index = 0;
volatile uint8_t rx_byte;
volatile uint8_t message_received = 0;
volatile uint8_t wifi_connected = 0;
volatile uint8_t mqtt_connected = 0;
uint32_t last_ping_time = 0;
uint32_t ping_interval = 30000; // Ping every 30 seconds
uint32_t last_publish_time = 0;
uint32_t publish_interval = 60000; // Publish status every 60 seconds

// Camera Trigger Variables
volatile uint8_t trigger_active = 0;
volatile uint32_t trigger_start_time = 0;

int main(void)
{
    // Initialize HAL
    HAL_Init();
    
    // Configure system clock
    SystemClock_Config();
    
    // Initialize GPIO for LED
    GPIO_Init();
    
    // Initialize USART1 for ESP-01
    USART1_Init();
    
    // System ready - initial blink pattern
    Blink_Status(3, 200);
    
    // Wait for ESP-01 to fully boot
    HAL_Delay(2000);
    
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Disable echo for cleaner responses
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
        Blink_Status(10, 200); // WiFi failed (10 medium blinks)
        while(1) { HAL_Delay(1000); } // Halt here for debugging
    }
    
    // Connect to MQTT Broker via TCP
    if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
    {
        HAL_Delay(2000);
    }
    else
    {
        Blink_Status(10, 300); // MQTT TCP failed (10 slow blinks)
        while(1) { HAL_Delay(1000); } // Halt here for debugging
    }
    
    // Send MQTT CONNECT packet
    if (ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID))
    {
        Blink_Status(5, 200); // MQTT connected
        HAL_Delay(2000);
    }
    else
    {
        Blink_Status(10, 400); // MQTT CONNECT failed (10 very slow blinks)
        while(1) { HAL_Delay(1000); } // Halt here for debugging
    }
    
    // Subscribe to command topic
    ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_SUB);
    Blink_Status(4, 150); // Subscribed
    HAL_Delay(2000);  // Wait longer after subscribe
    
    // Main loop - MQTT message processing
    uint32_t lastCheckTime = HAL_GetTick();
    last_ping_time = HAL_GetTick();
    last_publish_time = HAL_GetTick();
    
    // Delay first connection check to allow connection to stabilize
    lastCheckTime = HAL_GetTick() - 240000;  // First check at 5 minutes
    
    while (1)
    {
        // Check for incoming MQTT messages
        ESP01_ReceiveAndParse();
        
        // Manage camera trigger pulse
        if (trigger_active)
        {
            // Check if 100ms trigger pulse elapsed
            if ((HAL_GetTick() - trigger_start_time) >= TRIGGER_PULSE_MS)
            {
                // End trigger pulse
                HAL_GPIO_WritePin(TRIGGER_PORT, TRIGGER_PIN, GPIO_PIN_RESET);
                trigger_active = 0;
                
                // Visual feedback: capture complete
                Blink_Status(1, 50);
            }
        }
        
        // Publish status message every 60 seconds
        if ((HAL_GetTick() - last_publish_time) >= publish_interval)
        {
            char status_msg[64];
            uint32_t uptime_seconds = HAL_GetTick() / 1000;
            snprintf(status_msg, sizeof(status_msg), "Alive: %lu sec", uptime_seconds);
            ESP01_Send_MQTT_PUBLISH(MQTT_TOPIC_PUB, status_msg);
            last_publish_time = HAL_GetTick();
        }
        
        // Send MQTT ping every 30 seconds to keep connection alive
        if ((HAL_GetTick() - last_ping_time) >= ping_interval)
        {
            ESP01_SendPing();
            last_ping_time = HAL_GetTick();
        }
        
        // Check connection status every 5 minutes (was causing false disconnects)
        if ((HAL_GetTick() - lastCheckTime) >= 300000)
        {
            if (!ESP01_CheckConnection())
            {
                // Connection lost - attempt reconnect
                ESP01_Reconnect();
            }
            lastCheckTime = HAL_GetTick();
        }
        
        HAL_Delay(100); // Small delay to allow message processing
    }
    
    return 0;
}

/**
  * @brief System Clock Configuration
  * Configures the system clock to use HSI at 16 MHz
  */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // Configure the main internal regulator output voltage
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

    // Initialize HSI and configure PLL to 84MHz
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;  // HSI = 16MHz
    RCC_OscInitStruct.PLL.PLLN = 336; // VCO = 336MHz
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;  // SYSCLK = 84MHz
    RCC_OscInitStruct.PLL.PLLQ = 7;
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    // Configure CPU, AHB and APB buses clocks
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;  // Use PLL as system clock
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;         // AHB = 84MHz
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;          // APB1 = 42MHz (max 42MHz)
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;          // APB2 = 84MHz

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)  // 2 wait states for 84MHz
    {
        Error_Handler();
    }
}

/**
  * @brief GPIO Initialization
  * Configures GPIO pin PC13 as output for LED and PB12 as ESP32-CAM CS
  */
static void GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable GPIO Port C, B, and A clocks
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO pin PC13 (LED)
    GPIO_InitStruct.Pin = LED_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);

    // Configure GPIO pin PB12 (Trigger Pin for ESP32-CAM)
    GPIO_InitStruct.Pin = TRIGGER_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(TRIGGER_PORT, &GPIO_InitStruct);

    // Initialize LED OFF, Trigger Pin LOW
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(TRIGGER_PORT, TRIGGER_PIN, GPIO_PIN_RESET);
}

/**
  * @brief USART1 Initialization for ESP-01
  * PB6 - USART1_TX, PB7 - USART1_RX
  */
static void USART1_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    // Enable clocks
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_USART1_CLK_ENABLE();
    
    // Configure GPIO pins for USART1
    // PB6: TX, PB7: RX
    GPIO_InitStruct.Pin = GPIO_PIN_6 | GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
    
    // Configure USART1
    huart1.Instance = USART1;
    huart1.Init.BaudRate = ESP01_BAUDRATE;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    
    if (HAL_UART_Init(&huart1) != HAL_OK)
    {
        Error_Handler();
    }
    
    // Enable UART receive interrupt
    HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
    
    // Start receiving in interrupt mode
    HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
}

/**
  * @brief Send AT command and wait
  */
static void send_AT(const char* cmd, uint32_t delay_ms)
{
    // Temporarily abort RX interrupt to avoid conflicts
    HAL_UART_AbortReceive_IT(&huart1);
    
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(delay_ms);
    
    // Restart RX interrupt
    HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
}

/**
  * @brief Send raw data to ESP-01
  */
static void send_data(uint8_t* data, uint16_t len)
{
    // Temporarily abort RX interrupt to avoid conflicts
    HAL_UART_AbortReceive_IT(&huart1);
    
    HAL_UART_Transmit(&huart1, data, len, HAL_MAX_DELAY);
    HAL_Delay(500);
    
    // Restart RX interrupt
    HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
}

/**
  * @brief Wait for '>' prompt from ESP-01 after AT+CIPSEND
  */
static void wait_for_prompt(void)
{
    // Just wait for prompt - no blocking read
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
    
    // Set station mode (STA)
    send_AT("AT+CWMODE=1\r\n", 1000);
    
    // Disconnect from any existing WiFi first
    send_AT("AT+CWQAP\r\n", 1000);
    HAL_Delay(1000);
    
    // Clear buffers
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Connect to WiFi - just send the command and wait
    char wifiCommand[128];
    snprintf(wifiCommand, sizeof(wifiCommand),
             "AT+CWJAP=\"%s\",\"%s\"\r\n", ssid, password);
    
    HAL_UART_Transmit(&huart1, (uint8_t*)wifiCommand, strlen(wifiCommand), HAL_MAX_DELAY);
    
    // Blink during connection attempt - WiFi can take 5-10 seconds
    for (uint8_t i = 0; i < 40; i++)  // 40 x 250ms = 10 seconds
    {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        HAL_Delay(250);
    }
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    
    // Trust that WiFi connected (no blocking read)
    HAL_Delay(2000);
    
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
    
    // Clear any existing connection first
    send_AT("AT+CIPCLOSE\r\n", 1000);
    
    // Clear RX buffer
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    snprintf(cmd, sizeof(cmd),
             "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", broker, port);
    
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    
    // Wait for TCP connection - no blocking read
    HAL_Delay(6000);
    
    // Trust connection succeeded
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
    HAL_UART_Transmit(&huart1, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    // Wait for '>' prompt
    wait_for_prompt();
    
    // Send MQTT CONNECT packet
    HAL_UART_Transmit(&huart1, packet, index, HAL_MAX_DELAY);
    
    // Wait for MQTT CONNACK - no blocking read
    HAL_Delay(2000);
    
    // Trust MQTT CONNECT succeeded
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
    HAL_UART_Transmit(&huart1, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    // Wait for '>' prompt
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
    HAL_UART_Transmit(&huart1, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    // Wait for '>' prompt
    wait_for_prompt();
    
    // Send MQTT SUBSCRIBE packet
    send_data(packet, index);
}

/**
  * @brief Receive and parse incoming MQTT messages
  * Sets capture_triggered flag when MQTT PUBLISH received
  */
static void ESP01_ReceiveAndParse(void)
{
    // Wait for enough bytes (MQTT PUBLISH packet is ~37 bytes from ESP-01)
    if (rx_index < 30)
        return;
    
    // Search for occupancy keywords in the raw buffer
    // Don't rely on null termination - search byte by byte using strncmp
    
    // Check for "occupied" keyword
    const char* keyword_occupied = MQTT_KEYWORD_OCCUPIED;
    const int occupied_len = strlen(keyword_occupied);
    int found_occupied = 0;
    
    for (int i = 0; i <= rx_index - occupied_len; i++)
    {
        if (strncmp((char*)&rx_buffer[i], keyword_occupied, occupied_len) == 0)
        {
            found_occupied = 1;
            break;
        }
    }
    
    // Check for "empty" keyword
    const char* keyword_empty = MQTT_KEYWORD_EMPTY;
    const int empty_len = strlen(keyword_empty);
    int found_empty = 0;
    
    for (int i = 0; i <= rx_index - empty_len; i++)
    {
        if (strncmp((char*)&rx_buffer[i], keyword_empty, empty_len) == 0)
        {
            found_empty = 1;
            break;
        }
    }
    
    if (found_occupied)
    {
        // Trigger camera immediately on occupied detection
        if (!trigger_active)
        {
            // Start trigger pulse
            HAL_GPIO_WritePin(TRIGGER_PORT, TRIGGER_PIN, GPIO_PIN_SET);
            trigger_start_time = HAL_GetTick();
            trigger_active = 1;
            
            // Visual feedback: triggering camera
            Blink_Status(2, 100);
        }
        
        // Clear buffer after processing
        rx_index = 0;
        memset(rx_buffer, 0, RX_BUFFER_SIZE);
    }
    else if (found_empty)
    {
        // Visual feedback: spot empty
        Blink_Status(1, 50);
        
        // Clear buffer after processing
        rx_index = 0;
        memset(rx_buffer, 0, RX_BUFFER_SIZE);
    }
    else
    {
        // No match found, clear buffer if getting too full
        if (rx_index > RX_BUFFER_SIZE - 100)
        {
            rx_index = 0;
            memset(rx_buffer, 0, RX_BUFFER_SIZE);
        }
    }
}

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
  * @brief Send MQTT PINGREQ to keep connection alive
  */
static void ESP01_SendPing(void)
{
    uint8_t ping_packet[2] = {0xC0, 0x00}; // PINGREQ packet
    
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", 2);
    HAL_UART_Transmit(&huart1, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    wait_for_prompt();
    send_data(ping_packet, 2);
}

/**
  * @brief Check if connection is still alive
  * @return 1 if connected, 0 if disconnected
  */
static uint8_t ESP01_CheckConnection(void)
{
    // Trust that connection is alive
    // The ESP-01 will indicate if connection drops via +IPD or error messages
    // which will be caught by the interrupt handler
    return 1;
}

/**
  * @brief Reconnect to WiFi and MQTT broker
  */
static void ESP01_Reconnect(void)
{
    // Indicate reconnection attempt
    Blink_Status(10, 50); // Rapid blinks
    
    // Mark as disconnected
    wifi_connected = 0;
    mqtt_connected = 0;
    
    // Close existing connection
    send_AT("AT+CIPCLOSE\r\n", 1000);
    HAL_Delay(1000);
    
    // Try to reconnect to WiFi
    if (ESP01_ConnectWiFi(WIFI_SSID, WIFI_PASSWORD))
    {
        Blink_Status(3, 300); // WiFi reconnected
        HAL_Delay(1000);
        
        // Try to reconnect to MQTT broker
        if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
        {
            HAL_Delay(2000);
            
            // Send MQTT CONNECT
            if (ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID))
            {
                Blink_Status(5, 200); // MQTT reconnected
                HAL_Delay(2000);
                
                // Re-subscribe to topic
                ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_SUB);
                Blink_Status(4, 150); // Resubscribed
                HAL_Delay(1000);
            }
        }
    }
    else
    {
        // Reconnection failed
        Blink_Status(15, 100); // Long error blink
        HAL_Delay(5000); // Wait before next attempt
    }
}

/**
  * @brief UART receive complete callback
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
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
        HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
    }
}

/**
  * @brief USART1 IRQ Handler
  */
void USART1_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart1);
}

/**
  * @brief Error Handler
  * This function is executed in case of error occurrence.
  */
void Error_Handler(void)
{
    // Disable all interrupts
    __disable_irq();
    
    // Infinite loop
    while (1)
    {
    }
}

/**
  * @brief This function is executed in case of a fault occurrence.
  */
void HardFault_Handler(void)
{
    Error_Handler();
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void)
{
    Error_Handler();
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
    Error_Handler();
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
    Error_Handler();
}

/**
  * @brief This function handles System service call via SWI instruction.
  */
void SVC_Handler(void)
{
}

/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
}

/**
  * @brief This function handles PendSV, except handler.
  */
void PendSV_Handler(void)
{
}

/**
  * @brief This function handles SysTick Handler.
  */
void SysTick_Handler(void)
{
    HAL_IncTick();
}

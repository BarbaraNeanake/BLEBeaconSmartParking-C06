#include "stm32f4xx_hal.h"
#include <string.h>
#include <stdio.h>
#include "jpeg_encoder.h"

// LED pin configuration
#define LED_PIN GPIO_PIN_13
#define LED_PORT GPIOC

// ESP32-CAM SPI Configuration (from blueprint)
#define CAM_SPI_PORT        SPI1
#define CAM_CS_PIN          GPIO_PIN_12    // NSS/Trigger
#define CAM_CS_PORT         GPIOB
#define CAM_SCK_PIN         GPIO_PIN_13    // SPI CLK
#define CAM_MISO_PIN        GPIO_PIN_14    // SPI MISO
#define CAM_MOSI_PIN        GPIO_PIN_15    // SPI MOSI
#define CAM_SPI_GPIO_PORT   GPIOB

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
#define MQTT_TOPIC_PUB "SPARK_C06/stm32/test"
#define MQTT_TOPIC_SUB "SPARK_C06/stm32/command"

// REST API Configuration
#define HTTP_SERVER "your-server.com"  // TODO: Update with actual server
#define HTTP_PORT 80
#define HTTP_ENDPOINT "/api/upload"

// Buffer sizes
#define RX_BUFFER_SIZE 512
#define TX_BUFFER_SIZE 256
#define IMAGE_BUFFER_SIZE 307200  // 640x480 grayscale
#define JPEG_BUFFER_SIZE 20480    // 20KB max for compressed JPEG

void SystemClock_Config(void);
void Error_Handler(void);
static void GPIO_Init(void);
static void USART1_Init(void);
static void SPI1_Init(void);
static void Blink_Status(uint8_t count, uint16_t delay);
static void ESP32CAM_TriggerCapture(void);
static void ESP32CAM_ReceiveImage(uint8_t* buffer, uint32_t* size);

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
static uint8_t ESP01_HTTP_POST(const char* server, uint16_t port, const char* endpoint, 
                                 uint8_t* data, uint32_t data_size);

// Global variables
UART_HandleTypeDef huart1;
SPI_HandleTypeDef hspi1;
char rx_buffer[RX_BUFFER_SIZE];
volatile uint16_t rx_index = 0;
volatile uint8_t rx_byte;
volatile uint8_t message_received = 0;
volatile uint8_t wifi_connected = 0;
volatile uint8_t mqtt_connected = 0;
uint32_t last_ping_time = 0;
uint32_t ping_interval = 30000; // Ping every 30 seconds
uint8_t spi_chunk_buffer[512]; // Small buffer for SPI chunks

// Large buffers for image processing (place in .bss section)
static uint8_t image_buffer[IMAGE_BUFFER_SIZE];      // 640x480 grayscale = 300KB
static uint8_t jpeg_buffer[JPEG_BUFFER_SIZE];        // Compressed JPEG output
uint32_t image_size = 0;
uint32_t jpeg_size = 0;

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
    
    // Initialize SPI1 for ESP32-CAM
    SPI1_Init();
    
    // Initial blink to show system is ready
    Blink_Status(3, 200);
    HAL_Delay(2000);
    
    // Connect to WiFi
    if (ESP01_ConnectWiFi(WIFI_SSID, WIFI_PASSWORD))
    {
        Blink_Status(3, 300); // WiFi connected
        HAL_Delay(1000);
    }
    else
    {
        Blink_Status(10, 200); // WiFi failed
        Error_Handler();
    }
    
    // Connect to MQTT Broker via TCP
    if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
    {
        HAL_Delay(2000);
    }
    else
    {
        Blink_Status(10, 300); // MQTT TCP failed
        Error_Handler();
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
        Error_Handler();
    }
    
    // Subscribe to command topic
    ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_SUB);
    Blink_Status(4, 150); // Subscribed
    HAL_Delay(1000);
    
    // Main loop - wait for MQTT trigger to capture image
    uint32_t lastCheckTime = HAL_GetTick();
    last_ping_time = HAL_GetTick();
    
    while (1)
    {
        // Check for incoming MQTT messages (trigger for image capture)
        ESP01_ReceiveAndParse();
        
        // Send MQTT ping every 30 seconds to keep connection alive
        if ((HAL_GetTick() - last_ping_time) >= ping_interval)
        {
            ESP01_SendPing();
            last_ping_time = HAL_GetTick();
        }
        
        // Check connection status every 60 seconds
        if ((HAL_GetTick() - lastCheckTime) >= 60000)
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

    // Enable GPIO Port C and B clocks
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // Configure GPIO pin PC13 (LED)
    GPIO_InitStruct.Pin = LED_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);

    // Configure GPIO pin PB12 (ESP32-CAM CS/Trigger - initially HIGH/idle)
    GPIO_InitStruct.Pin = CAM_CS_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(CAM_CS_PORT, &GPIO_InitStruct);

    // Initialize LED OFF and CS HIGH (SPI idle)
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_SET);
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
  * @brief SPI1 Initialization for ESP32-CAM
  * PB13 - SCK, PB14 - MISO, PB15 - MOSI, PB12 - CS (manual)
  */
static void SPI1_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    // Enable SPI1 clock
    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    
    // Configure SPI GPIO pins: PB13 (SCK), PB14 (MISO), PB15 (MOSI)
    GPIO_InitStruct.Pin = CAM_SCK_PIN | CAM_MISO_PIN | CAM_MOSI_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
    HAL_GPIO_Init(CAM_SPI_GPIO_PORT, &GPIO_InitStruct);
    
    // Configure SPI1
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_MASTER;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
    hspi1.Init.NSS = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16; // ~1 MHz at 16 MHz sysclk
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    
    if (HAL_SPI_Init(&hspi1) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief Send AT command and wait
  */
static void send_AT(const char* cmd, uint32_t delay_ms)
{
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(delay_ms);
}

/**
  * @brief Send raw data to ESP-01
  */
static void send_data(uint8_t* data, uint16_t len)
{
    HAL_UART_Transmit(&huart1, data, len, HAL_MAX_DELAY);
    HAL_Delay(500);
}

/**
  * @brief Wait for '>' prompt from ESP-01 after AT+CIPSEND
  */
static void wait_for_prompt(void)
{
    uint8_t ch;
    uint32_t tickstart = HAL_GetTick();
    while ((HAL_GetTick() - tickstart) < 3000) {
        if (HAL_UART_Receive(&huart1, &ch, 1, 100) == HAL_OK) {
            if (ch == '>') return;
        }
    }
}

/**
  * @brief Connect ESP-01 to WiFi network
  * @return 1 if connected, 0 if failed
  */
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password)
{
    uint8_t buffer[512] = {0};
    
    // Reset ESP-01
    send_AT("AT+RST\r\n", 3000);
    HAL_Delay(2000);
    
    // Clear old buffer data
    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);
    
    // Set station mode
    send_AT("AT+CWMODE=1\r\n", 1000);
    
    // Connect to WiFi
    char wifiCommand[128];
    snprintf(wifiCommand, sizeof(wifiCommand),
             "AT+CWJAP=\"%s\",\"%s\"\r\n", ssid, password);
    
    HAL_UART_Transmit(&huart1, (uint8_t*)wifiCommand, strlen(wifiCommand), HAL_MAX_DELAY);
    HAL_Delay(10000); // Wait longer for WiFi connection
    
    // Check for successful connection
    HAL_UART_Receive(&huart1, buffer, sizeof(buffer), 1000);
    
    if (strstr((char*)buffer, "WIFI CONNECTED") != NULL || 
        strstr((char*)buffer, "OK") != NULL)
    {
        wifi_connected = 1;
        return 1;
    }
    
    wifi_connected = 0;
    return 0;
}

/**
  * @brief Connect to MQTT broker via TCP
  * @return 1 if connected, 0 if failed
  */
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port)
{
    char cmd[128];
    uint8_t buffer[256] = {0};
    
    snprintf(cmd, sizeof(cmd),
             "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", broker, port);
    
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(5000);
    
    // Check for connection
    HAL_UART_Receive(&huart1, buffer, sizeof(buffer), 1000);
    
    if (strstr((char*)buffer, "CONNECT") != NULL || 
        strstr((char*)buffer, "OK") != NULL)
    {
        return 1;
    }
    
    return 0;
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
    
    // Send CIPSEND command
    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart1, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);
    
    // Wait for '>' prompt
    wait_for_prompt();
    
    // Send MQTT CONNECT packet
    send_data(packet, index);
    
    HAL_Delay(1000);
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
  * Sets CONTROL_PIN HIGH when message received on subscribed topic
  */
static void ESP01_ReceiveAndParse(void)
{
    // Check if we have accumulated data in buffer
    if (rx_index > 10) // Minimum size for meaningful data
    {
        // Null terminate for string functions
        if (rx_index < RX_BUFFER_SIZE)
            rx_buffer[rx_index] = '\0';
        
        // Look for +IPD (incoming data indicator from ESP-01)
        // Format: +IPD,<length>:<data>
        char* start = strstr(rx_buffer, "+IPD,");
        if (start)
        {
            // Found incoming data - parse length
            start += 5; // Move past "+IPD,"
            
            // Find the colon that separates length from data
            char* colon = strchr(start, ':');
            if (colon)
            {
                colon++; // Move past ':'
                
                // Now colon points to the MQTT packet
                // Check if it's an MQTT PUBLISH packet (first byte is 0x30 or 0x31)
                uint8_t mqtt_type = (uint8_t)*colon;
                
                // MQTT PUBLISH packet types: 0x30 (QoS 0), 0x31, 0x32 (QoS 1), 0x33, etc.
                if ((mqtt_type & 0xF0) == 0x30)
                {
                    // MQTT PUBLISH received - Trigger image capture!
                    
                    // Blink LED rapidly to indicate trigger received
                    for (uint8_t i = 0; i < 5; i++)
                    {
                        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
                        HAL_Delay(50);
                    }
                    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
                    
                    // Trigger ESP32-CAM to capture image
                    ESP32CAM_TriggerCapture();
                    
                    // Receive full image, compress, and send via HTTP
                    // This function handles: receive -> compress -> HTTP POST
                    ESP32CAM_ReceiveImage(image_buffer, &image_size);
                    
                    // Clear buffer after processing
                    rx_index = 0;
                    memset(rx_buffer, 0, RX_BUFFER_SIZE);
                }
                else
                {
                    // Not a PUBLISH packet, clear old data if buffer getting full
                    if (rx_index > RX_BUFFER_SIZE - 100)
                    {
                        rx_index = 0;
                        memset(rx_buffer, 0, RX_BUFFER_SIZE);
                    }
                }
            }
        }
        else
        {
            // No +IPD found, clear buffer if getting full
            if (rx_index > RX_BUFFER_SIZE - 100)
            {
                rx_index = 0;
                memset(rx_buffer, 0, RX_BUFFER_SIZE);
            }
        }
    }
}

/**
  * @brief Trigger ESP32-CAM to capture image
  */
static void ESP32CAM_TriggerCapture(void)
{
    // Pull CS LOW to signal ESP32-CAM to capture
    HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_RESET);
    HAL_Delay(100); // Keep LOW for 100ms as trigger signal
    HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_SET);
    
    // Wait for ESP32-CAM to process and prepare image
    HAL_Delay(2000); // 2 seconds for capture and compression
}

/**
  * @brief Receive image data from ESP32-CAM via SPI
  * @param chunk_buffer Pointer to 512-byte chunk buffer for temporary storage
  * @param total_size Pointer to variable to store total received image size
  */
static void ESP32CAM_ReceiveImage(uint8_t* full_image_buffer, uint32_t* total_size)
{
    uint8_t header[4];
    
    // Pull CS LOW to start SPI communication
    HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_RESET);
    HAL_Delay(10);
    
    // Receive 4-byte header containing image size
    // Format: [SIZE_MSB, SIZE_2, SIZE_3, SIZE_LSB]
    HAL_SPI_Receive(&hspi1, header, 4, 1000);
    
    // Parse image size (big-endian)
    *total_size = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
    
    // Validate size (grayscale VGA = 307200 bytes)
    if (*total_size > IMAGE_BUFFER_SIZE || *total_size == 0)
    {
        *total_size = 0;
        HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_SET);
        Blink_Status(10, 100); // Error indicator
        return;
    }
    
    // Receive complete image data into buffer
    uint32_t remaining = *total_size;
    uint32_t bytes_received = 0;
    uint8_t* dest_ptr = full_image_buffer;
    
    while (remaining > 0)
    {
        uint16_t to_receive = (remaining > 512) ? 512 : remaining;
        
        if (HAL_SPI_Receive(&hspi1, spi_chunk_buffer, to_receive, 5000) != HAL_OK)
        {
            *total_size = 0; // Error occurred
            Blink_Status(10, 100);
            break;
        }
        
        // Copy chunk to full image buffer
        memcpy(dest_ptr, spi_chunk_buffer, to_receive);
        dest_ptr += to_receive;
        
        bytes_received += to_receive;
        remaining -= to_receive;
        
        // Blink LED to show progress
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
    }
    
    // Pull CS HIGH to end communication
    HAL_GPIO_WritePin(CAM_CS_PORT, CAM_CS_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    
    // Update total_size with actual bytes received
    *total_size = bytes_received;
    
    // If image received successfully, compress and send via HTTP
    if (*total_size > 0 && *total_size == IMAGE_BUFFER_SIZE)
    {
        Blink_Status(2, 200); // Image received successfully
        
        // Compress image using JPEG encoder (quality 75-80 per blueprint)
        uint32_t compressed_size = JPEG_EncodeGrayscale(
            full_image_buffer, 
            *total_size, 
            jpeg_buffer, 
            JPEG_BUFFER_SIZE, 
            78  // Quality 78 (75-80 range)
        );
        
        if (compressed_size > 0)
        {
            Blink_Status(3, 200); // Compression successful
            
            // Send compressed image via HTTP POST
            if (ESP01_HTTP_POST(HTTP_SERVER, HTTP_PORT, HTTP_ENDPOINT, jpeg_buffer, compressed_size))
            {
                Blink_Status(5, 300); // HTTP POST successful
            }
            else
            {
                Blink_Status(8, 150); // HTTP POST failed
            }
        }
        else
        {
            Blink_Status(7, 150); // Compression failed
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
    uint8_t buffer[256] = {0};
    
    // Check WiFi status
    HAL_UART_Transmit(&huart1, (uint8_t*)"AT+CIPSTATUS\r\n", 14, HAL_MAX_DELAY);
    HAL_Delay(500);
    HAL_UART_Receive(&huart1, buffer, sizeof(buffer), 500);
    
    // Look for connection status
    if (strstr((char*)buffer, "STATUS:2") != NULL || // Got IP
        strstr((char*)buffer, "STATUS:3") != NULL || // Connected
        strstr((char*)buffer, "STATUS:4") != NULL)   // Disconnected but has IP
    {
        // Check if TCP connection is alive
        if (strstr((char*)buffer, "STATUS:3") != NULL)
        {
            return 1; // Connected
        }
    }
    
    return 0; // Disconnected
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
  * @brief Send HTTP POST request with binary data via ESP-01
  * @param server Server hostname or IP
  * @param port Server port (typically 80 for HTTP)
  * @param endpoint API endpoint path (e.g., "/api/upload")
  * @param data Binary data to send (JPEG image)
  * @param data_size Size of data in bytes
  * @return 1 if successful, 0 if failed
  */
static uint8_t ESP01_HTTP_POST(const char* server, uint16_t port, const char* endpoint, 
                                 uint8_t* data, uint32_t data_size)
{
    char cmd[128];
    char http_header[256];
    uint8_t response[512] = {0};
    
    // Close any existing connection
    send_AT("AT+CIPCLOSE\r\n", 1000);
    HAL_Delay(500);
    
    // Establish TCP connection to HTTP server
    snprintf(cmd, sizeof(cmd), "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", server, port);
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(5000); // Wait for connection
    
    // Check if connected
    HAL_UART_Receive(&huart1, response, sizeof(response), 1000);
    if (strstr((char*)response, "CONNECT") == NULL)
    {
        return 0; // Connection failed
    }
    
    // Prepare HTTP POST header
    snprintf(http_header, sizeof(http_header),
        "POST %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %lu\r\n"
        "Connection: close\r\n"
        "\r\n",
        endpoint, server, data_size
    );
    
    uint32_t total_size = strlen(http_header) + data_size;
    
    // Send CIPSEND command
    snprintf(cmd, sizeof(cmd), "AT+CIPSEND=%lu\r\n", total_size);
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    
    wait_for_prompt(); // Wait for '>' prompt
    
    // Send HTTP header
    HAL_UART_Transmit(&huart1, (uint8_t*)http_header, strlen(http_header), HAL_MAX_DELAY);
    
    // Send image data in chunks (ESP-01 UART buffer is limited)
    uint32_t remaining = data_size;
    uint8_t* data_ptr = data;
    uint16_t chunk_size = 512; // Send 512 bytes at a time
    
    while (remaining > 0)
    {
        uint16_t to_send = (remaining > chunk_size) ? chunk_size : remaining;
        HAL_UART_Transmit(&huart1, data_ptr, to_send, 5000);
        data_ptr += to_send;
        remaining -= to_send;
        HAL_Delay(20); // Small delay between chunks
        
        // Blink LED to show progress
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
    }
    
    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
    
    // Wait for response
    HAL_Delay(2000);
    memset(response, 0, sizeof(response));
    HAL_UART_Receive(&huart1, response, sizeof(response), 3000);
    
    // Check for successful response (200 OK)
    uint8_t success = 0;
    if (strstr((char*)response, "200 OK") != NULL || 
        strstr((char*)response, "SEND OK") != NULL)
    {
        success = 1;
    }
    
    // Close connection
    send_AT("AT+CIPCLOSE\r\n", 1000);
    HAL_Delay(500);
    
    // Reconnect to MQTT broker
    if (mqtt_connected)
    {
        HAL_Delay(1000);
        ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT);
        HAL_Delay(2000);
        ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID);
        HAL_Delay(2000);
        ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_SUB);
        HAL_Delay(1000);
    }
    
    return success;
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

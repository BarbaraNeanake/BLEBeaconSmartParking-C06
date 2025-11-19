/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : VL53L0X + MQTT Integration for STM32F103C8T6
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "vl53l0x.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// VL53L0X Constants
#define MEASUREMENT_INTERVAL_MS  200
#define DISTANCE_THRESHOLD_NEAR  100
#define DISTANCE_THRESHOLD_FAR   1000

// LED Configuration
#define LED_PIN GPIO_PIN_13
#define LED_PORT GPIOC

// Buzzer Configuration
#define BUZZER_PIN GPIO_PIN_14
#define BUZZER_PORT GPIOC

// WiFi Credentials
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"

// MQTT Broker Configuration
#define MQTT_BROKER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_CLIENT_ID "SPARK_C06"

// MQTT Topics
#define MQTT_SLOT_ID "1"
#define MQTT_TOPIC_BASE "SPARK_C06/isOccupied/"
#define MQTT_TOPIC_PING "SPARK_C06/ping/"
#define MQTT_TOPIC_SLOT MQTT_TOPIC_BASE MQTT_SLOT_ID
#define MQTT_TOPIC_ALARM MQTT_TOPIC_PING MQTT_SLOT_ID  // Subscribe to alarm
#define MQTT_KEYWORD_OCCUPIED "True"
#define MQTT_KEYWORD_EMPTY "False"

// Buffer sizes
#define RX_BUFFER_SIZE 512

// Alarm Time
#define ALARM_DURATION_MS 5000              // 5 seconds alarm duration
#define BUZZER_TOGGLE_INTERVAL_MS 200       // Toggle buzzer every 200ms

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// VL53L0X Variables
VL53L0X_RangingData_t rangingData;
VL53L0X_DeviceInfo_t deviceInfo;
uint16_t distance_mm = 0;
uint32_t measurementCount = 0;

// MQTT Variables
char rx_buffer[RX_BUFFER_SIZE];
volatile uint16_t rx_index = 0;
volatile uint8_t rx_byte;
volatile uint8_t wifi_connected = 0;
volatile uint8_t mqtt_connected = 0;
uint32_t last_ping_time = 0;
uint32_t last_publish_time = 0;

// Parking State Variables
uint8_t spot_occupied = 0;  // 0 = empty, 1 = occupied
uint8_t last_spot_state = 0;

// Alarm Variables
volatile uint8_t alarm_active = 0;          // Whether alarm is currently active
volatile uint32_t alarm_start_time = 0;     // When the alarm started
volatile uint8_t buzzer_state = 0;          // Current state of buzzer (0=off, 1=on)

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_I2C1_Init(void);
/* USER CODE BEGIN PFP */
// VL53L0X Functions
void VL53L0X_ProcessRangingData(VL53L0X_RangingData_t *data);
const char* b;

// ESP-01 MQTT Functions
static void send_AT(const char* cmd, uint32_t delay_ms);
static void send_data(uint8_t* data, uint16_t len);
static void wait_for_prompt(void);
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password);
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port);
static uint8_t ESP01_Send_MQTT_CONNECT(const char* clientID);
static void ESP01_Send_MQTT_PUBLISH(const char* topic, const char* message);
void ESP01_Send_MQTT_SUBSCRIBE(const char *topic, uint16_t packet_id);
static void ESP01_SendPing(void);
static void Blink_Status(uint8_t count, uint16_t delay);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
  * @brief  Retargets printf to UART2
  */
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart1, (uint8_t *)ptr, len, HAL_MAX_DELAY);
    return len;
}

/**
  * @brief  Process and display complete ranging data
  */
void VL53L0X_ProcessRangingData(VL53L0X_RangingData_t *data)
{
    printf("[#%lu] Distance: %4d mm", measurementCount++, data->range_mm);

    if (data->range_status == 0) {
        printf(" | %.2f cm | Signal: %3d",
               data->range_mm / 10.0f,
               data->signal_rate);

        // Visual indicator
        if (data->range_mm < DISTANCE_THRESHOLD_NEAR) {
            printf(" [VERY CLOSE]");
        } else if (data->range_mm < DISTANCE_THRESHOLD_FAR) {
            printf(" [NEAR]");
        } else {
            printf(" [FAR]");
        }
        printf(" ✓");
    } else {
        printf(" | Status: %s", VL53L0X_GetRangeStatusString(data->range_status));
    }
    printf("\r\n");
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

void send_and_receive(const char *cmd, uint32_t delay_ms) {
    uint8_t buffer[512] = {0};
    if (cmd) HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(delay_ms);
    /* Try read (timeout kecil biar nggak nge-blok terlalu lama) */
    if (HAL_UART_Receive(&huart2, buffer, sizeof(buffer)-1, 200) == HAL_OK) {
        buffer[sizeof(buffer)-1] = 0;
        printf("ESP_RESP: %s\r\n", buffer);
    }
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
  */
static uint8_t ESP01_ConnectWiFi(const char* ssid, const char* password)
{
    printf("Connecting to WiFi: %s\r\n", ssid);

    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);

    send_AT("AT+CWMODE=1\r\n", 1000);
    send_AT("AT+CWQAP\r\n", 1000);
    HAL_Delay(1000);

    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);

    char wifiCommand[128];
    snprintf(wifiCommand, sizeof(wifiCommand),
             "AT+CWJAP=\"%s\",\"%s\"\r\n", ssid, password);

    HAL_UART_Transmit(&huart2, (uint8_t*)wifiCommand, strlen(wifiCommand), HAL_MAX_DELAY);

    // Wait for response instead of blocking delay
    uint32_t start_time = HAL_GetTick();
    uint8_t connected = 0;
    const uint32_t timeout = 15000;  // 15 seconds timeout

    printf("Waiting for WiFi response");
    while ((HAL_GetTick() - start_time) < timeout) {
        if (rx_index > 0) {
            if (strstr(rx_buffer, "WIFI GOT IP") ||
                strstr(rx_buffer, "OK")) {
                connected = 1;
                break;
            }
            else if (strstr(rx_buffer, "FAIL") ||
                     strstr(rx_buffer, "ERROR")) {
                connected = 0;
                break;
            }
            // Clear buffer after checking
            rx_index = 0;
            memset(rx_buffer, 0, RX_BUFFER_SIZE);
        }
        HAL_Delay(100);
        printf(".");
    }
    printf("\r\n");

    HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);  // Ensure LED off after attempt

    if (connected) {
        wifi_connected = 1;
        printf("WiFi connected!\r\n");
        return 1;
    } else {
        wifi_connected = 0;
        printf("ERROR: WiFi connection failed! Last response: %s\r\n", rx_buffer);
        return 0;
    }
}

/**
  * @brief Connect to MQTT broker via TCP
  */
static uint8_t ESP01_ConnectMQTTBroker(const char* broker, uint16_t port)
{
    printf("Connecting to MQTT broker: %s:%d\r\n", broker, port);

    char cmd[128];

    send_AT("AT+CIPCLOSE\r\n", 1000);

    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);

    snprintf(cmd, sizeof(cmd),
             "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", broker, port);

    HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
    HAL_Delay(6000);

    printf("TCP connection established!\r\n");
    return 1;
}

/**
  * @brief Send MQTT CONNECT packet
  */
static uint8_t ESP01_Send_MQTT_CONNECT(const char* clientID)
{
    printf("Sending MQTT CONNECT...\r\n");

    uint8_t packet[128];
    uint8_t clientID_len = strlen(clientID);
    uint8_t total_len = 10 + 2 + clientID_len;

    uint8_t index = 0;
    packet[index++] = 0x10;
    packet[index++] = total_len;
    packet[index++] = 0x00; packet[index++] = 0x04;
    packet[index++] = 'M'; packet[index++] = 'Q';
    packet[index++] = 'T'; packet[index++] = 'T';
    packet[index++] = 0x04;
    packet[index++] = 0x02;
    packet[index++] = 0x00; packet[index++] = 0x3C;
    packet[index++] = 0x00; packet[index++] = clientID_len;
    memcpy(&packet[index], clientID, clientID_len);
    index += clientID_len;

    rx_index = 0;
    memset(rx_buffer, 0, RX_BUFFER_SIZE);

    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);

    wait_for_prompt();

    HAL_UART_Transmit(&huart2, packet, index, HAL_MAX_DELAY);

    // Wait for CONNACK response instead of blocking delay
    uint32_t start_time = HAL_GetTick();
    uint8_t connected = 0;
    const uint32_t timeout = 5000;  // 5 seconds timeout

    printf("Waiting for MQTT CONNACK");
    while ((HAL_GetTick() - start_time) < timeout) {
        if (rx_index > 0) {
            // Check for CONNACK packet (first byte 0x20) or successful ESP response
            if ((rx_index >= 4 && rx_buffer[0] == 0x20 && rx_buffer[3] == 0x00) ||
                strstr(rx_buffer, "OK") ||
                strstr(rx_buffer, "SEND OK")) {
                connected = 1;
                break;
            }
            else if (strstr(rx_buffer, "FAIL") ||
                     strstr(rx_buffer, "ERROR") ||
                     (rx_index >= 4 && rx_buffer[0] == 0x20 && rx_buffer[3] != 0x00)) {
                connected = 0;
                break;
            }
            // Clear buffer after checking to prevent overflow
            rx_index = 0;
            memset(rx_buffer, 0, RX_BUFFER_SIZE);
        }
        HAL_Delay(100);
        printf(".");
    }
    printf("\r\n");

    if (connected) {
        mqtt_connected = 1;
        printf("✅ MQTT connected successfully!\r\n");
        return 1;
    } else {
        mqtt_connected = 0;
        printf("❌ MQTT connection FAILED! Response: %s\r\n", rx_buffer[0] ? rx_buffer : "No response");
        return 0;
    }
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

    packet[index++] = 0x30;
    packet[index++] = remaining_len;
    packet[index++] = 0x00;
    packet[index++] = topic_len;
    memcpy(&packet[index], topic, topic_len);
    index += topic_len;
    memcpy(&packet[index], message, message_len);
    index += message_len;

    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", index);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);

    wait_for_prompt();
    send_data(packet, index);

    printf("Published: %s -> %s\r\n", topic, message);
}

void ESP01_Send_MQTT_SUBSCRIBE(const char *topic, uint16_t packet_id) {
    if (!topic) return;
    printf("Subscribing to topic: %s\r\n", topic);

        uint16_t tlen = strlen(topic);
        uint16_t rem_len = 2 + 2 + tlen + 1; // Packet ID + topic length + topic + QoS

        uint8_t pkt[64];
        uint16_t idx = 0;

        // Fixed header (SUBSCRIBE packet type with QoS 1)
        pkt[idx++] = 0x82;  // MQTT SUBSCRIBE packet type
        pkt[idx++] = (uint8_t)rem_len;

        // Packet Identifier
        pkt[idx++] = (uint8_t)((packet_id >> 8) & 0xFF);
        pkt[idx++] = (uint8_t)(packet_id & 0xFF);

        // Topic length
        pkt[idx++] = (uint8_t)((tlen >> 8) & 0xFF);
        pkt[idx++] = (uint8_t)(tlen & 0xFF);

        // Topic string
        memcpy(&pkt[idx], topic, tlen);
        idx += tlen;

        // QoS level (0)
        pkt[idx++] = 0x00;

        // Send command to ESP
        char sendCmd[32];
        snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%u\r\n", idx);

        // Use consistent communication method
        send_AT(sendCmd, 1000);
        send_data(pkt, idx);

        printf("Subscription packet sent\r\n");

        // Wait for SUBACK response
        HAL_Delay(1000);
        printf("ESP Response Buffer: %s\r\n", rx_buffer); // Debug what we received
}

void process_incoming_messages(void)
{
	// Look for ESP-01 data notification: "+IPD,<len>:<data>"
	    char *ipd_ptr = strstr(rx_buffer, "+IPD");
	    if (ipd_ptr) {
	        // Extract length and data
	        uint16_t data_len = 0;
	        char *len_ptr = strchr(ipd_ptr, ',');
	        if (len_ptr && sscanf(len_ptr+1, "%hu", &data_len) == 1) {
	            char *data_ptr = strchr(ipd_ptr, ':');
	            if (data_ptr && data_len > 0) {
	                data_ptr++; // Skip the colon

	                // Look for MQTT PUBLISH packet (first byte 0x30)
	                if ((uint8_t)data_ptr[0] == 0x30) {
	                    // Parse topic and message
	                    uint8_t topic_len = (data_ptr[2] << 8) | data_ptr[3];
	                    char topic[64] = {0};
	                    char message[64] = {0};

	                    // Extract topic
	                    if (topic_len < sizeof(topic)) {
	                        strncpy(topic, &data_ptr[4], topic_len);
	                        topic[topic_len] = '\0';
	                    }

	                    // Extract message (simplified)
	                    uint8_t msg_len = data_len - 4 - topic_len;
	                    if (msg_len > 0 && msg_len < sizeof(message)) {
	                        strncpy(message, &data_ptr[4 + topic_len], msg_len);
	                        message[msg_len] = '\0';
	                    }

	                    // Handle specific topics - ACTIVATE BUZZER ALARM
	                    if (strstr(topic, MQTT_TOPIC_ALARM)) {
	                        printf("🚨 ALARM ACTIVATED VIA MQTT!\r\n");
	                        printf("Topic: %s | Message: %s\r\n", topic, message);

	                        // Start 5-second buzzer alarm
	                        __disable_irq();
	                        alarm_active = 1;
	                        alarm_start_time = HAL_GetTick();
	                        buzzer_state = 0; // Start with buzzer off
	                        __enable_irq();

	                        // Turn on LED as visual indicator
	                        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_SET);
	                    }
	                }
	            }
	        }

	        // Clear buffer after processing
	        rx_index = 0;
	        memset(rx_buffer, 0, RX_BUFFER_SIZE);
	    }
}

void handle_buzzer_alarm(void)
{
    if (!alarm_active) {
        // Ensure buzzer is off when no alarm
        if (buzzer_state != 0) {
            HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
            buzzer_state = 0;
        }
        return;
    }

    uint32_t current_time = HAL_GetTick();
    uint32_t elapsed_time = current_time - alarm_start_time;

    // Check if alarm duration has elapsed
    if (elapsed_time >= ALARM_DURATION_MS) {
        // Deactivate alarm
        alarm_active = 0;
        HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(LED_PORT, LED_PIN, GPIO_PIN_RESET);
        buzzer_state = 0;
        printf("✅ ALARM DEACTIVATED (5 seconds elapsed)\r\n");
        return;
    }

    // Toggle buzzer at defined interval
    static uint32_t last_toggle_time = 0;
    if ((current_time - last_toggle_time) >= BUZZER_TOGGLE_INTERVAL_MS) {
        last_toggle_time = current_time;

        // Toggle buzzer state
        if (buzzer_state == 0) {
            HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN, GPIO_PIN_SET);
            buzzer_state = 1;
            printf("🔊 BUZZER ON [%lu/%lu ms]\r\n", elapsed_time, ALARM_DURATION_MS);
        } else {
            HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN, GPIO_PIN_RESET);
            buzzer_state = 0;
            printf("🔇 BUZZER OFF [%lu/%lu ms]\r\n", elapsed_time, ALARM_DURATION_MS);
        }
    }
}

/**
  * @brief Send MQTT PINGREQ to keep connection alive
  */
static void ESP01_SendPing(void)
{
    uint8_t ping_packet[2] = {0xC0, 0x00};

    char sendCmd[32];
    snprintf(sendCmd, sizeof(sendCmd), "AT+CIPSEND=%d\r\n", 2);
    HAL_UART_Transmit(&huart2, (uint8_t*)sendCmd, strlen(sendCmd), HAL_MAX_DELAY);

    wait_for_prompt();
    send_data(ping_packet, 2);
}

/**
  * @brief UART receive complete callback
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        if (rx_index < RX_BUFFER_SIZE - 1)
        {
            rx_buffer[rx_index++] = rx_byte;
            rx_buffer[rx_index] = '\0'; // Null-terminate for string operations
        }
        else
        {
            rx_index = 0;
            memset(rx_buffer, 0, RX_BUFFER_SIZE);
        }

        HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
    }
}

/**
  * @brief  GPIO EXTI Callback - VL53L0X interrupt (optional)
  */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_0) {
        if (VL53L0X_ReadRangeData(&hi2c1, &rangingData) == VL53L0X_OK) {
            VL53L0X_ClearInterrupt(&hi2c1);
        }
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  VL53L0X_Status_t vl_status;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_USART1_UART_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */

  printf("\r\n\r\n");
  printf("====================================================\r\n");
  printf("   SPARK - SMART PARKING SOLUTION - C06               \r\n");
  printf("====================================================\r\n\r\n");

  // System ready blink
  HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1);
  Blink_Status(3, 200);
  HAL_Delay(2000);

  // Clear RX buffer
  rx_index = 0;
  memset(rx_buffer, 0, RX_BUFFER_SIZE);


  // Disable echo
  send_AT("ATE0\r\n", 500);
  HAL_Delay(500);

  // Connect to WiFi
  while(!wifi_connected) {
	  ESP01_ConnectWiFi(WIFI_SSID, WIFI_PASSWORD);
  }

  // Connect to MQTT Broker
  if (ESP01_ConnectMQTTBroker(MQTT_BROKER, MQTT_PORT))
  {
      HAL_Delay(2000);
  }
  else
  {
      printf("MQTT broker connection failed!\r\n");
      Blink_Status(10, 300);
      while(1) { HAL_Delay(1000); }
  }

  // Send MQTT CONNECT
  if (ESP01_Send_MQTT_CONNECT(MQTT_CLIENT_ID))
  {
      Blink_Status(5, 200);
      HAL_Delay(2000);
  }
  else
  {
      printf("MQTT CONNECT failed!\r\n");
      Blink_Status(10, 400);
      while(1) { HAL_Delay(1000); }
  }

  printf("\r\n====================================================\r\n");
  printf("MQTT Setup Complete! Starting VL53L0X...\r\n");
  printf("====================================================\r\n\r\n");




  // Initialize VL53L0X
  printf("Initializing VL53L0X sensor...\r\n");
  while (!VL53L0X_IsDeviceReady(&hi2c1)) {
      printf("ERROR: VL53L0X not detected!\r\n");
      printf("Retrying to find VL53L0X sensor...\r\n");
      VL53L0X_IsDeviceReady(&hi2c1);
      HAL_Delay(500);
  }

  vl_status = VL53L0X_Init(&hi2c1);
  while (vl_status != VL53L0X_OK) {
      printf("ERROR: VL53L0X initialization failed!\r\n");
      printf("Retrying VL53L0X initialization...\r\n");
      vl_status = VL53L0X_Init(&hi2c1);
      HAL_Delay(500);
  }
  printf("VL53L0X initialized successfully!\r\n\r\n");

  // Get device info
  if (VL53L0X_GetDeviceInfo(&hi2c1, &deviceInfo) == VL53L0X_OK) {
      printf("Model ID: 0x%02X | Revision: 0x%02X\r\n\r\n",
             deviceInfo.model_id, deviceInfo.revision_id);
  }

  // Configure sensor
  VL53L0X_SetProfile(&hi2c1, VL53L0X_PROFILE_DEFAULT);
  VL53L0X_SetMeasurementTimingBudget(&hi2c1, 33000);
  VL53L0X_ConfigureInterrupt(&hi2c1, true);

  // Start continuous measurement
  vl_status = VL53L0X_StartMeasurement(&hi2c1, VL53L0X_MODE_CONTINUOUS);
  if (vl_status != VL53L0X_OK) {
      printf("ERROR: Failed to start measurement!\r\n");
      while(1);
  }

  printf("System ready! Monitoring parking spot...\r\n");
  printf("====================================================\r\n\r\n");

  send_AT("AT\r\n", 1000);
  printf("AT Response: %s\r\n", rx_buffer);
  ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_ALARM, 1);
  HAL_Delay(2000);
  printf("After subscribe buffer: %s\r\n", rx_buffer);

  HAL_Delay(200);

  last_ping_time = HAL_GetTick();
  last_publish_time = HAL_GetTick();
  ESP01_Send_MQTT_SUBSCRIBE(MQTT_TOPIC_ALARM, 1);
  //ESP01_Send_MQTT_PUBLISH(MQTT_TOPIC_SLOT, "BANG TOLONG BANG");
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // Handle buzzer alarm if active
	      handle_buzzer_alarm();

	      // Process any incoming MQTT messages
	      if (rx_index > 0) {
	          process_incoming_messages();
	      }

	      HAL_Delay(10); // Small delay for responsiveness


    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // Read VL53L0X sensor

    vl_status = VL53L0X_ReadRangeData(&hi2c1, &rangingData);

    if (vl_status == VL53L0X_OK && rangingData.range_status == 0)
    {
        VL53L0X_ProcessRangingData(&rangingData);

        // Determine if spot is occupied (car detected within threshold)
        if (rangingData.range_mm < DISTANCE_THRESHOLD_FAR) {
            // Car detected
            if (!spot_occupied) {
                spot_occupied = 1;
                printf("\r\n>>> CAR DETECTED! Publishing to MQTT...\r\n");
                ESP01_Send_MQTT_PUBLISH(MQTT_TOPIC_SLOT, MQTT_KEYWORD_OCCUPIED);
                Blink_Status(3, 100);
                printf("\r\n");
            }
        } else {
            // No car - spot empty
            if (spot_occupied) {
                spot_occupied = 0;
                printf("\r\n>>> SPOT CLEARED! Publishing to MQTT...\r\n");
                ESP01_Send_MQTT_PUBLISH(MQTT_TOPIC_SLOT, MQTT_KEYWORD_EMPTY);
                Blink_Status(2, 150);
                printf("\r\n");
            }
        }
    }

    VL53L0X_ClearInterrupt(&hi2c1);

    // Send MQTT ping every 30 seconds
    if ((HAL_GetTick() - last_ping_time) >= 30000)
    {
        ESP01_SendPing();
        last_ping_time = HAL_GetTick();
        printf("MQTT ping sent\r\n");
    }

    // Periodic status update every 60 seconds
    if ((HAL_GetTick() - last_publish_time) >= 60000)
    {
        char status_msg[64];
        snprintf(status_msg, sizeof(status_msg), "Spot: %s | Count: %lu",
                 spot_occupied ? "OCCUPIED" : "EMPTY", measurementCount);
        ESP01_Send_MQTT_PUBLISH("SPARK_C06/status", status_msg);
        last_publish_time = HAL_GetTick();
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
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
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
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
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_14, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC14 */
  GPIO_InitStruct.Pin = GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
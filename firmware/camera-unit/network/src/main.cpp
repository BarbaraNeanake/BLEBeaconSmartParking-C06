#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "driver/gpio.h"
#include "mqtt_client.h"
// This header contains the function for attaching the certificate bundle
#include "esp_crt_bundle.h"

// =================== Configuration ===================
// --- Wi-Fi Credentials
#define WIFI_SSID      "Waifai"
#define WIFI_PASSWORD  "123654987"

// --- HiveMQ Cloud MQTT Broker Details
// The URI format handles the secure connection (mqtts://) and port
#define MQTT_BROKER_URI "mqtts://3831b88a3b374f0ca37fea8fa4ff4ff2.s1.eu.hivemq.cloud:8883"
#define MQTT_USERNAME   "CAM_UNIT"
#define MQTT_PASSWORD   "CAPStone12345"

// --- Device & Topic Configuration
#define MQTT_CLIENT_ID    "esp32-c3-device-a-idf"
#define MQTT_TOPIC_DATA   "sensor/1/data"
#define MQTT_TOPIC_ALARM  "violation/1" // New topic for the buzzer

// --- Hardware Pin Configuration
#define BUTTON_PIN GPIO_NUM_4
#define BUZZER_PIN GPIO_NUM_5 // GPIO for the active buzzer
// =====================================================

static const char *TAG = "DEVICE_A";

// Global handle for the MQTT client
static esp_mqtt_client_handle_t mqtt_client;

// Global state variable, volatile as it's modified in a task and read by others
static volatile bool isActive = false;

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from Wi-Fi, retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP address: " IPSTR, IP2STR(&event->ip_info.ip));
        // Start the MQTT client once we have an IP
        esp_mqtt_client_start(mqtt_client);
    }
}

void wifi_init_sta(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASSWORD);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%ld", base, event_id);
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    esp_mqtt_client_handle_t client = event->client;

    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
        // Subscribe to the violation topic upon connection
        esp_mqtt_client_subscribe(client, MQTT_TOPIC_ALARM, 0);
        ESP_LOGI(TAG, "Subscribed to topic: %s", MQTT_TOPIC_ALARM);
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
        break;
    case MQTT_EVENT_DATA: // Event for incoming messages on subscribed topics
        ESP_LOGI(TAG, "MQTT_EVENT_DATA");
        // Check if the received message is on the alarm topic
        if (strncmp(event->topic, MQTT_TOPIC_ALARM, event->topic_len) == 0) {
            // Check if the payload is "True"
            if (strncmp(event->data, "True", event->data_len) == 0) {
                ESP_LOGW(TAG, "VIOLATION DETECTED! Sounding buzzer.");
                gpio_set_level(BUZZER_PIN, 1); // Turn buzzer ON
            } else {
                ESP_LOGI(TAG, "Alarm cleared. Turning buzzer OFF.");
                gpio_set_level(BUZZER_PIN, 0); // Turn buzzer OFF
            }
        }
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT_EVENT_ERROR");
        if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
            ESP_LOGE(TAG, "Last error code reported from esp-tls: 0x%x", event->error_handle->esp_tls_last_esp_err);
        }
        break;
    default:
        ESP_LOGI(TAG, "Other event id:%d", event->event_id);
        break;
    }
}

static void mqtt_app_start(void) {
    esp_mqtt_client_config_t mqtt_cfg = {};
    
    mqtt_cfg.broker.address.uri = MQTT_BROKER_URI;
    mqtt_cfg.credentials.username = MQTT_USERNAME;
    mqtt_cfg.credentials.authentication.password = MQTT_PASSWORD;
    mqtt_cfg.credentials.client_id = MQTT_CLIENT_ID;
    mqtt_cfg.broker.verification.crt_bundle_attach = esp_crt_bundle_attach;

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(mqtt_client, (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
}

void button_task(void *pvParameter) {
    ESP_LOGI(TAG, "Button task started. Watching GPIO %d", BUTTON_PIN);
    
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << BUTTON_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);
    
    int last_state = 1; // 1 = HIGH (not pressed), 0 = LOW (pressed)

    while (1) {
        int current_state = gpio_get_level(BUTTON_PIN);
        
        if (last_state == 1 && current_state == 0) {
            vTaskDelay(pdMS_TO_TICKS(50));
            current_state = gpio_get_level(BUTTON_PIN);
            if (current_state == 0) {
                isActive = !isActive;
                const char* payload = isActive ? "True" : "False";

                ESP_LOGI(TAG, "Button pressed! New state: %s", payload);
                
                int msg_id = esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC_DATA, payload, 0, 1, 1);
                if(msg_id != -1) {
                    ESP_LOGI(TAG, "Message published successfully, msg_id=%d", msg_id);
                } else {
                    ESP_LOGE(TAG, "Failed to publish message.");
                }
            }
        }
        
        last_state = current_state;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "ESP-IDF Device A booting...");
    
    // Configure the buzzer pin as an output
    gpio_config_t buzzer_io_conf = {};
    buzzer_io_conf.intr_type = GPIO_INTR_DISABLE;
    buzzer_io_conf.mode = GPIO_MODE_OUTPUT;
    buzzer_io_conf.pin_bit_mask = (1ULL << BUZZER_PIN);
    buzzer_io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    buzzer_io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&buzzer_io_conf);
    gpio_set_level(BUZZER_PIN, 0); // Ensure buzzer is off on startup

    mqtt_app_start();
    wifi_init_sta();

    xTaskCreate(button_task, "button_task", 2048, NULL, 10, NULL);
}


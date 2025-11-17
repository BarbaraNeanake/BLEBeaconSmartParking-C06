#include <stdio.h>
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "mqtt_client.h"
#include "lwip/err.h"
#include "lwip/sys.h"

// WiFi credentials (replace with yours)
#define WIFI_SSID "Waifai"
#define WIFI_PASS "123654987"

// MQTT config for HiveMQ
#define MQTT_BROKER_URI "mqtt://3831b88a3b374f0ca37fea8fa4ff4ff2.s1.eu.hivemq.cloud:8883"  // Public HiveMQ; change if private
#define MQTT_USERNAME "CAM_UNIT"             // Replace with your username
#define MQTT_PASSWORD "Spark12345"             // Replace with your password
#define MQTT_TOPIC "capture"                       // Topic to subscribe to
#define MQTT_TRIGGER_PAYLOAD "snap"                      // Payload to trigger capture (e.g., publish this via MQTT client)

static const char *TAG = "ESP32-CAM";
static EventGroupHandle_t s_wifi_event_group;
static EventGroupHandle_t mqtt_event_group;
const int WIFI_CONNECTED_BIT = BIT0;
const int MQTT_CONNECTED_BIT = BIT0;

static esp_mqtt_client_handle_t mqtt_client;  // Global to avoid unused warning

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void capture_task(void *arg)
{
    ESP_LOGI(TAG, "Capturing image...");

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera capture failed");
        vTaskDelete(NULL);
        return;
    }

    // Image captured to RAM buffer (fb->buf). Do something with it here if needed
    // (e.g., publish base64 over MQTT, send via HTTP, etc.). For now, just log size.
    ESP_LOGI(TAG, "Image captured: %u bytes (format: %s, size: %ux%u)",
             fb->len, (fb->format == PIXFORMAT_JPEG ? "JPEG" : "Unknown"),
             fb->width, fb->height);

    // Release frame buffer
    esp_camera_fb_return(fb);

    ESP_LOGI(TAG, "Capture complete");
    vTaskDelete(NULL);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = event_data;
    esp_mqtt_client_handle_t client = event->client;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected to %s", MQTT_BROKER_URI);
            xEventGroupSetBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            esp_mqtt_client_subscribe(client, MQTT_TOPIC, 0);
            ESP_LOGI(TAG, "Subscribed to topic: %s", MQTT_TOPIC);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected");
            xEventGroupClearBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            break;
        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT message received on topic: %.*s", event->topic_len, event->topic);
            ESP_LOGI(TAG, "Payload: %.*s", event->data_len, event->data);

            // Check for trigger payload
            if (event->data_len == strlen(MQTT_TRIGGER_PAYLOAD) &&
                strncmp((char *)event->data, MQTT_TRIGGER_PAYLOAD, event->data_len) == 0) {
                ESP_LOGI(TAG, "Trigger payload detected - capturing image...");
                xTaskCreate(capture_task, "capture_task", 4096, NULL, 5, NULL);
            }
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGI(TAG, "MQTT event error");
            break;
        default:
            ESP_LOGI(TAG, "Other MQTT event id:%d", event_id);
            break;
    }
}

static void wifi_init(void)
{
    ESP_LOGI(TAG, "Starting WiFi initialization...");
    
    ESP_LOGI(TAG, "Initializing network interface...");
    ESP_ERROR_CHECK(esp_netif_init());
    
    ESP_LOGI(TAG, "Creating event loop...");
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    ESP_LOGI(TAG, "Creating WiFi station interface...");
    esp_netif_create_default_wifi_sta();

    ESP_LOGI(TAG, "Configuring WiFi driver...");
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_LOGI(TAG, "Registering event handlers...");
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    ESP_LOGI(TAG, "Setting WiFi mode to STA...");
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    
    ESP_LOGI(TAG, "Starting WiFi...");
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi initialized, connecting to SSID: %s", WIFI_SSID);
}

static esp_mqtt_client_handle_t mqtt_client_init(void)
{
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URI,
        .credentials.username = MQTT_USERNAME,
        .credentials.authentication.password = MQTT_PASSWORD,
    };

    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(client);

    return client;
}

// Camera pin definitions for AI-Thinker ESP32-CAM (OV2640)
#define CAM_PIN_PWDN        32
#define CAM_PIN_RESET       -1  // Software reset
#define CAM_PIN_XCLK        0
#define CAM_PIN_SIOD        26
#define CAM_PIN_SIOC        27

#define CAM_PIN_D7          35
#define CAM_PIN_D6          34
#define CAM_PIN_D5          39
#define CAM_PIN_D4          36
#define CAM_PIN_D3          21
#define CAM_PIN_D2          19
#define CAM_PIN_D1          18
#define CAM_PIN_D0          5
#define CAM_PIN_VSYNC       25
#define CAM_PIN_HREF        23
#define CAM_PIN_PCLK        22

static esp_err_t init_camera(void)
{
    camera_config_t config = {
        .pin_pwdn       = CAM_PIN_PWDN,
        .pin_reset      = CAM_PIN_RESET,
        .pin_xclk       = CAM_PIN_XCLK,
        .pin_sccb_sda   = CAM_PIN_SIOD,
        .pin_sccb_scl   = CAM_PIN_SIOC,

        .pin_d7         = CAM_PIN_D7,
        .pin_d6         = CAM_PIN_D6,
        .pin_d5         = CAM_PIN_D5,
        .pin_d4         = CAM_PIN_D4,
        .pin_d3         = CAM_PIN_D3,
        .pin_d2         = CAM_PIN_D2,
        .pin_d1         = CAM_PIN_D1,
        .pin_d0         = CAM_PIN_D0,
        .pin_vsync      = CAM_PIN_VSYNC,
        .pin_href       = CAM_PIN_HREF,
        .pin_pclk       = CAM_PIN_PCLK,

        .xclk_freq_hz   = 20000000,
        .ledc_timer     = LEDC_TIMER_0,
        .ledc_channel   = LEDC_CHANNEL_0,

        .pixel_format   = PIXFORMAT_JPEG,  // JPEG output
        .frame_size     = FRAMESIZE_UXGA,  // 1600x1200 (adjust if needed; lower for less RAM use)
        .jpeg_quality   = 12,              // 10-63 (lower = higher quality)
        .fb_count       = 1,               // 1 frame buffer (enough for single capture)
        .grab_mode      = CAMERA_GRAB_WHEN_EMPTY,
        .fb_location    = CAMERA_FB_IN_PSRAM,  // Use PSRAM for buffer
    };

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return err;
    }

    ESP_LOGI(TAG, "Camera initialized successfully");
    return ESP_OK;
}

void app_main(void)
{
    // Add delay to allow serial monitor to connect
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    printf("\n\n");
    printf("===========================================\n");
    printf("ESP32-CAM Starting...\n");
    printf("===========================================\n");
    printf("Chip: ESP32\n");
    printf("Free heap: %lu bytes\n", esp_get_free_heap_size());
    printf("===========================================\n\n");
    
    ESP_LOGI(TAG, "*** ESP32-CAM Application Starting ***");
    ESP_LOGI(TAG, "Free heap at start: %lu bytes", esp_get_free_heap_size());
    
    // Initialize NVS
    ESP_LOGI(TAG, "Initializing NVS...");
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGI(TAG, "NVS partition was truncated, erasing...");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "NVS initialized successfully");

    // Create event groups
    ESP_LOGI(TAG, "Creating event groups...");
    s_wifi_event_group = xEventGroupCreate();
    mqtt_event_group = xEventGroupCreate();
    ESP_LOGI(TAG, "Event groups created");

    // Init WiFi
    ESP_LOGI(TAG, "Initializing WiFi...");
    wifi_init();

    // Wait for WiFi connection
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi connected successfully!");

    // Init camera
    ESP_LOGI(TAG, "Initializing camera...");
    esp_err_t cam_err = init_camera();
    if (cam_err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization FAILED! Error: 0x%x", cam_err);
        ESP_LOGE(TAG, "System will continue but camera will not work");
    } else {
        ESP_LOGI(TAG, "Camera initialized successfully!");
    }

    // Init MQTT
    ESP_LOGI(TAG, "Initializing MQTT client...");
    mqtt_client = mqtt_client_init();
    ESP_LOGI(TAG, "MQTT client started");

    ESP_LOGI(TAG, "===========================================");
    ESP_LOGI(TAG, "Setup complete!");
    ESP_LOGI(TAG, "Publish '%s' to topic '%s' to trigger capture.", MQTT_TRIGGER_PAYLOAD, MQTT_TOPIC);
    ESP_LOGI(TAG, "Monitor at 115200 baud.");
    ESP_LOGI(TAG, "===========================================");
    
    printf("\n>>> System ready and running <<<\n\n");

    // Wait forever with periodic heartbeat
    uint32_t loop_count = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        loop_count++;
        ESP_LOGI(TAG, "Heartbeat #%lu - Free heap: %lu bytes", loop_count, esp_get_free_heap_size());
    }
}
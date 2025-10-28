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
#include "mqtt_client.h"
#include "esp_crt_bundle.h"
extern "C" {
    #include "esp_camera.h"
}
#include "esp_http_client.h"

// Configuration
#define WIFI_SSID "Waifai"
#define WIFI_PASSWORD "123654987"
#define MQTT_BROKER_URI "mqtts://3831b88a3b374f0ca37fea8fa4ff4ff2.s1.eu.hivemq.cloud:8883"
#define MQTT_USERNAME "CAM_UNIT"
#define MQTT_PASSWORD "CAPStone12345"
#define MQTT_CLIENT_ID "esp32-cam-device-idf"
#define MQTT_TRIGGER_TOPIC "/parking/trigger"
#define MQTT_STATUS_TOPIC "/camera/status"
#define HTTPS_URL "https://danishritonga-spark-backend.hf.space/upload"

static const char *TAG = "ESP32_CAM";

// Global handle for MQTT client
static esp_mqtt_client_handle_t mqtt_client;

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from Wi-Fi, retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP address: " IPSTR, IP2STR(&event->ip_info.ip));
        esp_mqtt_client_start(mqtt_client);
    }
}

static void wifi_init_sta(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASSWORD);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_MAX_MODEM));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi initialized.");
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(mqtt_client, MQTT_TRIGGER_TOPIC, 1);
        esp_mqtt_client_subscribe(mqtt_client, MQTT_STATUS_TOPIC, 1);
        ESP_LOGI(TAG, "Subscribed to %s", MQTT_TRIGGER_TOPIC);
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT disconnected, retrying...");
        break;
    case MQTT_EVENT_DATA:
        ESP_LOGI(TAG, "MQTT data received");
        if (strncmp(event->topic, MQTT_TRIGGER_TOPIC, event->topic_len) == 0) {
            ESP_LOGI(TAG, "Received trigger on %s", MQTT_TRIGGER_TOPIC);
            xTaskCreate(https_task, "https_task", 4096, NULL, 4, NULL);
        }
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT error");
        if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
            ESP_LOGE(TAG, "TLS error: 0x%x", event->error_handle->esp_tls_last_esp_err);
        }
        break;
    default:
        ESP_LOGI(TAG, "Other MQTT event id:%d", event->event_id);
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
    mqtt_cfg.keepalive = 3600;

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
}

static void https_task(void *pvParameter) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
        esp_mqtt_client_publish(mqtt_client, MQTT_STATUS_TOPIC, "uploading", 0, 1, 0);
        esp_http_client_config_t config = { .url = HTTPS_URL, .cert_pem = nullptr };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "image/jpeg");
        esp_http_client_set_post_field(client, fb->buf, fb->len);
        for (int i = 0; i < 3; ++i) {
            esp_err_t err = esp_http_client_perform(client);
            if (err == ESP_OK) break;
            vTaskDelay((1 << i) * 1000 / portTICK_PERIOD_MS);
        }
        esp_mqtt_client_publish(mqtt_client, MQTT_STATUS_TOPIC, "idle", 0, 1, 0);
        esp_http_client_cleanup(client);
        esp_camera_fb_return(fb);
    } else {
        esp_mqtt_client_publish(mqtt_client, MQTT_STATUS_TOPIC, "error", 0, 1, 0);
    }
    esp_deep_sleep_start();
    vTaskDelete(NULL);
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "ESP32-CAM booting...");

    // Initialize camera
    camera_config_t camera_config = {
        .pin_sccb_sda = 26, .pin_sccb_scl = 27,
        .pin_d7 = 35, .pin_d6 = 34, .pin_d5 = 39, .pin_d4 = 36,
        .pin_d3 = 21, .pin_d2 = 19, .pin_d1 = 18, .pin_d0 = 5,
        .pin_vsync = 25, .pin_href = 23, .pin_pclk = 22,
        .xclk_freq_hz = 20000000, .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0, .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QVGA, .jpeg_quality = 15, .fb_count = 1
    };
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        esp_mqtt_client_publish(mqtt_client, MQTT_STATUS_TOPIC, "camera_init_failed", 0, 1, 0);
        return;
    }

    mqtt_app_start();
    wifi_init_sta();
}
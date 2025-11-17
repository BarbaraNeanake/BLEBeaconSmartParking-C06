#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "sensor.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/i2c.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_http_client.h"
#include "esp_netif.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"

// WiFi credentials
#define WIFI_SSID "Waifai"
#define WIFI_PASS "123654987"

// Backend Configuration
#define BACKEND_URL "https://webhook.site/3b14c2f5-71b0-47e3-a9b1-d82b641c0e41"

// Trigger Pin (Blueprint: 100ms HIGH pulse from STM32 PB12)
#define TRIGGER_PIN 15  // GPIO15
#define DEBOUNCE_TIME_MS 500  // Ignore triggers within 500ms of last one

static const char *TAG = "ESP32CAM_TRIGGER";
static volatile bool trigger_detected = false;
static volatile uint32_t last_trigger_time = 0;

// ESP32-CAM AI-Thinker Pin Definition
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1 // Software reset will be performed
#define CAM_PIN_XCLK    0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27

#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0      5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

static camera_config_t camera_config = {
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
    
    .xclk_freq_hz   = 20000000,        // 20MHz
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,
    
    .pixel_format   = PIXFORMAT_JPEG,       // JPEG for blueprint
    .frame_size     = FRAMESIZE_VGA,        // VGA (640×480) as per blueprint
    .jpeg_quality   = 12,                   // Quality 12 = ~75-80%
    .fb_count       = 1,                    // Number of frame buffers
    .fb_location    = CAMERA_FB_IN_DRAM,    // Use DRAM
    .grab_mode      = CAMERA_GRAB_WHEN_EMPTY
};

const char* get_camera_model_name(camera_model_t model) {
    switch(model) {
        case CAMERA_OV7725:   return "OV7725";
        case CAMERA_OV2640:   return "OV2640";
        case CAMERA_OV3660:   return "OV3660";
        case CAMERA_OV5640:   return "OV5640";
        case CAMERA_OV7670:   return "OV7670";
        case CAMERA_NT99141:  return "NT99141";
        case CAMERA_GC2145:   return "GC2145";
        case CAMERA_GC032A:   return "GC032A";
        case CAMERA_GC0308:   return "GC0308";
        case CAMERA_BF3005:   return "BF3005";
        case CAMERA_BF20A6:   return "BF20A6";
        case CAMERA_SC101IOT: return "SC101IOT";
        case CAMERA_SC030IOT: return "SC030IOT";
        case CAMERA_SC031GS:  return "SC031GS";
        default:              return "UNKNOWN";
    }
}

const char* get_pixel_format_name(pixformat_t format) {
    switch(format) {
        case PIXFORMAT_RGB565:    return "RGB565";
        case PIXFORMAT_YUV422:    return "YUV422";
        case PIXFORMAT_YUV420:    return "YUV420";
        case PIXFORMAT_GRAYSCALE: return "GRAYSCALE";
        case PIXFORMAT_JPEG:      return "JPEG";
        case PIXFORMAT_RGB888:    return "RGB888";
        case PIXFORMAT_RAW:       return "RAW";
        case PIXFORMAT_RGB444:    return "RGB444";
        case PIXFORMAT_RGB555:    return "RGB555";
        default:                  return "UNKNOWN";
    }
}

const char* get_frame_size_name(framesize_t size) {
    switch(size) {
        case FRAMESIZE_96X96:    return "96x96";
        case FRAMESIZE_QQVGA:    return "QQVGA (160x120)";
        case FRAMESIZE_QCIF:     return "QCIF (176x144)";
        case FRAMESIZE_HQVGA:    return "HQVGA (240x176)";
        case FRAMESIZE_240X240:  return "240x240";
        case FRAMESIZE_QVGA:     return "QVGA (320x240)";
        case FRAMESIZE_CIF:      return "CIF (400x296)";
        case FRAMESIZE_HVGA:     return "HVGA (480x320)";
        case FRAMESIZE_VGA:      return "VGA (640x480)";
        case FRAMESIZE_SVGA:     return "SVGA (800x600)";
        case FRAMESIZE_XGA:      return "XGA (1024x768)";
        case FRAMESIZE_HD:       return "HD (1280x720)";
        case FRAMESIZE_SXGA:     return "SXGA (1280x1024)";
        case FRAMESIZE_UXGA:     return "UXGA (1600x1200)";
        case FRAMESIZE_FHD:      return "FHD (1920x1080)";
        default:                 return "UNKNOWN";
    }
}

void print_system_info(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "ESP32-CAM TRIGGER UNIT (Blueprint v1.0)");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Chip: %s", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free internal heap: %lu bytes", esp_get_free_internal_heap_size());
    
    // Check for PSRAM using heap capabilities
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "PSRAM found: %s", psram_size > 0 ? "YES" : "NO");
    if (psram_size > 0) {
        ESP_LOGI(TAG, "PSRAM size: %lu bytes", psram_size);
        ESP_LOGI(TAG, "Free PSRAM: %lu bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }
    ESP_LOGI(TAG, "========================================");
}

void print_camera_config(void) {
    ESP_LOGI(TAG, "Camera Configuration:");
    ESP_LOGI(TAG, "  XCLK Freq: %d Hz", camera_config.xclk_freq_hz);
    ESP_LOGI(TAG, "  Pixel Format: %s", get_pixel_format_name(camera_config.pixel_format));
    ESP_LOGI(TAG, "  Frame Size: %s", get_frame_size_name(camera_config.frame_size));
    ESP_LOGI(TAG, "  JPEG Quality: %d (0-63, lower=better)", camera_config.jpeg_quality);
    ESP_LOGI(TAG, "  Frame Buffers: %d", camera_config.fb_count);
    ESP_LOGI(TAG, "  FB Location: %s", camera_config.fb_location == CAMERA_FB_IN_PSRAM ? "PSRAM" : "DRAM");
}

void prepare_camera_pins(void) {
    ESP_LOGI(TAG, "Preparing camera pins...");
    
    // Power down the camera first
    if (CAM_PIN_PWDN != -1) {
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << CAM_PIN_PWDN),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE
        };
        gpio_config(&io_conf);
        
        // Power down camera (active high)
        gpio_set_level(CAM_PIN_PWDN, 1);
        vTaskDelay(pdMS_TO_TICKS(100));
        
        // Power up camera
        gpio_set_level(CAM_PIN_PWDN, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        
        ESP_LOGI(TAG, "Camera power cycled successfully");
    }
}

void scan_i2c_bus(void) {
    ESP_LOGI(TAG, "Scanning I2C bus...");
    
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CAM_PIN_SIOD,
        .scl_io_num = CAM_PIN_SIOC,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 100000,
    };
    
    esp_err_t err = i2c_param_config(I2C_NUM_0, &conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C config failed: %s", esp_err_to_name(err));
        return;
    }
    
    err = i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(err));
        return;
    }
    
    ESP_LOGI(TAG, "I2C initialized. Scanning for devices...");
    int devices_found = 0;
    
    for (uint8_t addr = 1; addr < 127; addr++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);
        
        esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
        i2c_cmd_link_delete(cmd);
        
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Found I2C device at address: 0x%02X", addr);
            devices_found++;
            
            // If this is the OV2640 address, try to read sensor ID
            if (addr == 0x30) {
                ESP_LOGI(TAG, "Device at 0x30 detected - attempting to read sensor ID registers...");
                
                // Try to read OV2640 ID registers
                // Register 0x0A (MIDH), 0x0B (MIDL), 0x1C (PID), 0x1D (VER)
                uint8_t reg_addr;
                uint8_t data;
                
                // Read Manufacturer ID High (should be 0x7F for OV2640)
                reg_addr = 0x0A;
                cmd = i2c_cmd_link_create();
                i2c_master_start(cmd);
                i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
                i2c_master_write_byte(cmd, reg_addr, true);
                i2c_master_start(cmd);
                i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_READ, true);
                i2c_master_read_byte(cmd, &data, I2C_MASTER_NACK);
                i2c_master_stop(cmd);
                ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
                i2c_cmd_link_delete(cmd);
                
                if (ret == ESP_OK) {
                    ESP_LOGI(TAG, "  MIDH (0x0A): 0x%02X (expected 0x7F for OV2640)", data);
                } else {
                    ESP_LOGW(TAG, "  Failed to read MIDH register");
                }
                
                // Read Product ID (should be 0x26 for OV2640)
                reg_addr = 0x1C;
                cmd = i2c_cmd_link_create();
                i2c_master_start(cmd);
                i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
                i2c_master_write_byte(cmd, reg_addr, true);
                i2c_master_start(cmd);
                i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_READ, true);
                i2c_master_read_byte(cmd, &data, I2C_MASTER_NACK);
                i2c_master_stop(cmd);
                ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
                i2c_cmd_link_delete(cmd);
                
                if (ret == ESP_OK) {
                    ESP_LOGI(TAG, "  PID (0x1C): 0x%02X (expected 0x26 for OV2640)", data);
                } else {
                    ESP_LOGW(TAG, "  Failed to read PID register");
                }
            }
        }
    }
    
    if (devices_found == 0) {
        ESP_LOGW(TAG, "No I2C devices found! Check wiring.");
    } else {
        ESP_LOGI(TAG, "Found %d I2C device(s)", devices_found);
    }
    
    // Remove I2C driver for camera to reinitialize
    i2c_driver_delete(I2C_NUM_0);
}

esp_err_t init_ledc_for_camera(void) {
    ESP_LOGI(TAG, "Initializing LEDC for camera clock...");
    
    ledc_timer_config_t ledc_timer = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = LEDC_TIMER_1_BIT,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = 20000000,  // 20MHz for camera
        .clk_cfg          = LEDC_AUTO_CLK
    };
    
    esp_err_t err = ledc_timer_config(&ledc_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LEDC timer config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    ledc_channel_config_t ledc_channel = {
        .gpio_num       = CAM_PIN_XCLK,
        .speed_mode     = LEDC_LOW_SPEED_MODE,
        .channel        = LEDC_CHANNEL_0,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = LEDC_TIMER_0,
        .duty           = 1,
        .hpoint         = 0
    };
    
    err = ledc_channel_config(&ledc_channel);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LEDC channel config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "LEDC initialized successfully");
    return ESP_OK;
}

// WiFi initialization (doesn't connect yet)
esp_err_t init_wifi(void) {
    ESP_LOGI(TAG, "Initializing WiFi...");
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_LOGI(TAG, "WiFi initialized (not connected yet)");
    
    return ESP_OK;
}

// Connect to WiFi (on-demand)
esp_err_t connect_wifi(void) {
    ESP_LOGI(TAG, "Connecting to WiFi: %s", WIFI_SSID);
    
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());
    
    // Wait for connection (simplified - check for IP address)
    int retry = 0;
    while (retry < 20) {  // 10 seconds max
        esp_netif_ip_info_t ip_info;
        esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
        
        if (netif && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
            if (ip_info.ip.addr != 0) {
                ESP_LOGI(TAG, "WiFi connected! IP: " IPSTR, IP2STR(&ip_info.ip));
                return ESP_OK;
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(500));
        retry++;
    }
    
    ESP_LOGE(TAG, "WiFi connection timeout");
    return ESP_FAIL;
}

// Disconnect WiFi (save power)
void disconnect_wifi(void) {
    esp_wifi_disconnect();
    esp_wifi_stop();
    ESP_LOGI(TAG, "WiFi disconnected");
}

// Upload image to backend via HTTP POST
esp_err_t upload_image(camera_fb_t *fb) {
    ESP_LOGI(TAG, "Uploading image to backend...");
    
    esp_http_client_config_t config = {
        .url = BACKEND_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 30000,
        .transport_type = HTTP_TRANSPORT_OVER_SSL,
        .skip_cert_common_name_check = true,
        .crt_bundle_attach = NULL,
    };
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    
    esp_http_client_set_header(client, "Content-Type", "image/jpeg");
    esp_http_client_set_post_field(client, (char*)fb->buf, fb->len);
    
    esp_err_t err = esp_http_client_perform(client);
    
    if (err == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        int content_length = esp_http_client_get_content_length(client);
        ESP_LOGI(TAG, "Upload complete: HTTP %d, Size: %d bytes, Response: %d bytes", 
                 status, fb->len, content_length);
    } else {
        ESP_LOGE(TAG, "Upload failed: %s", esp_err_to_name(err));
    }
    
    esp_http_client_cleanup(client);
    return err;
}

// Camera power management
void camera_power_on(void) {
    if (CAM_PIN_PWDN != -1) {
        gpio_set_level(CAM_PIN_PWDN, 0);  // Active LOW - power ON
        vTaskDelay(pdMS_TO_TICKS(150));  // Longer delay for sensor to stabilize
        
        // Restart LEDC clock for camera
        ledc_timer_resume(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0);
        
        ESP_LOGI(TAG, "Camera powered on and ready");
    }
}

void camera_power_off(void) {
    if (CAM_PIN_PWDN != -1) {
        // Pause LEDC clock to save power
        ledc_timer_pause(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0);
        
        gpio_set_level(CAM_PIN_PWDN, 1);  // Active LOW - power OFF
        ESP_LOGI(TAG, "Camera powered off");
    }
}

// GPIO ISR for trigger detection (rising edge = 100ms HIGH pulse from STM32)
static void IRAM_ATTR trigger_isr_handler(void* arg) {
    // Debounce: ignore triggers that happen too quickly
    uint32_t now = xTaskGetTickCountFromISR() * portTICK_PERIOD_MS;
    if ((now - last_trigger_time) < DEBOUNCE_TIME_MS) {
        return;  // Ignore this trigger (too soon after last one)
    }
    
    last_trigger_time = now;
    trigger_detected = true;
}

// Initialize trigger GPIO
esp_err_t init_trigger_gpio(void) {
    ESP_LOGI(TAG, "Initializing trigger GPIO%d...", TRIGGER_PIN);
    
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << TRIGGER_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_POSEDGE  // Trigger on rising edge (100ms HIGH pulse)
    };
    
    esp_err_t ret = gpio_config(&io_conf);
    if (ret == ESP_OK) {
        gpio_install_isr_service(0);
        gpio_isr_handler_add(TRIGGER_PIN, trigger_isr_handler, NULL);
        ESP_LOGI(TAG, "Trigger GPIO%d ready (rising edge)", TRIGGER_PIN);
    } else {
        ESP_LOGE(TAG, "Trigger GPIO config failed");
    }
    
    return ret;
}

// Process trigger: capture + upload workflow
void process_trigger(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "TRIGGER DETECTED - Starting workflow");
    ESP_LOGI(TAG, "========================================");
    
    uint32_t start_time = esp_timer_get_time() / 1000;  // ms
    
    // 1. Capture VGA JPEG (640×480) - camera stays powered
    ESP_LOGI(TAG, "Step 1: Capturing image...");
    
    // Discard first frame (may be stale from previous capture)
    camera_fb_t *fb_discard = esp_camera_fb_get();
    if (fb_discard) {
        esp_camera_fb_return(fb_discard);
    }
    
    // Capture fresh frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera capture failed!");
        return;
    }
    
    uint32_t capture_time = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "Captured: %d bytes (%dx%d JPEG) in %lu ms", 
             fb->len, fb->width, fb->height, capture_time - start_time);
    
    // Small delay to let power stabilize before WiFi
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // 2. Connect to WiFi
    ESP_LOGI(TAG, "Step 2: Connecting to WiFi...");
    if (connect_wifi() != ESP_OK) {
        ESP_LOGE(TAG, "WiFi connection failed!");
        esp_camera_fb_return(fb);
        return;
    }
    
    uint32_t wifi_time = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "WiFi connected in %lu ms", wifi_time - capture_time);
    
    // 3. Upload via HTTP POST
    ESP_LOGI(TAG, "Step 3: Uploading to backend...");
    esp_err_t upload_result = upload_image(fb);
    
    uint32_t upload_time = esp_timer_get_time() / 1000;
    if (upload_result == ESP_OK) {
        ESP_LOGI(TAG, "Upload completed in %lu ms", upload_time - wifi_time);
    }
    
    // 4. Cleanup
    esp_camera_fb_return(fb);
    disconnect_wifi();
    
    uint32_t total_time = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "WORKFLOW COMPLETE");
    ESP_LOGI(TAG, "Total time: %lu ms", total_time - start_time);
    ESP_LOGI(TAG, "  Capture: %lu ms", capture_time - start_time);
    ESP_LOGI(TAG, "  WiFi:    %lu ms", wifi_time - capture_time);
    ESP_LOGI(TAG, "  Upload:  %lu ms", upload_time - wifi_time);
    ESP_LOGI(TAG, "========================================");
}

esp_err_t init_camera(void) {
    ESP_LOGI(TAG, "Initializing camera...");
    
    // Prepare camera power pin
    prepare_camera_pins();
    
    // Scan I2C bus first
    scan_i2c_bus();
    
    // Initialize LEDC before camera
    esp_err_t err = init_ledc_for_camera();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LEDC initialization failed, continuing anyway...");
    }
    
    err = esp_camera_init(&camera_config);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        
        // Provide detailed error messages
        switch(err) {
            case ESP_ERR_CAMERA_NOT_DETECTED:
                ESP_LOGE(TAG, "Camera not detected. Check wiring and I2C connection.");
                ESP_LOGE(TAG, "Troubleshooting steps:");
                ESP_LOGE(TAG, "  1. Verify camera module is OV2640");
                ESP_LOGE(TAG, "  2. Check camera ribbon cable connection");
                ESP_LOGE(TAG, "  3. Verify I2C pins: SDA=GPIO%d, SCL=GPIO%d", CAM_PIN_SIOD, CAM_PIN_SIOC);
                ESP_LOGE(TAG, "  4. Check power supply (camera needs stable 3.3V)");
                ESP_LOGE(TAG, "  5. Try toggling PWDN pin (GPIO%d)", CAM_PIN_PWDN);
                break;
            case ESP_ERR_CAMERA_FAILED_TO_SET_FRAME_SIZE:
                ESP_LOGE(TAG, "Failed to set frame size.");
                break;
            case ESP_ERR_CAMERA_FAILED_TO_SET_OUT_FORMAT:
                ESP_LOGE(TAG, "Failed to set output format.");
                break;
            case ESP_ERR_CAMERA_NOT_SUPPORTED:
                ESP_LOGE(TAG, "Camera not supported on this chip.");
                ESP_LOGE(TAG, "Camera was detected at I2C address but failed sensor ID verification.");
                ESP_LOGE(TAG, "Possible causes:");
                ESP_LOGE(TAG, "  1. Camera sensor cannot be read (communication issue)");
                ESP_LOGE(TAG, "  2. Sensor is not OV2640 or not in supported sensor list");
                ESP_LOGE(TAG, "  3. Sensor needs more power or proper reset sequence");
                ESP_LOGE(TAG, "  4. PSRAM required but not enabled/detected");
                break;
            case ESP_ERR_NO_MEM:
                ESP_LOGE(TAG, "Not enough memory. Try reducing frame size or buffer count.");
                break;
            case ESP_ERR_NOT_FOUND:
                ESP_LOGE(TAG, "Camera sensor not found on I2C bus.");
                ESP_LOGE(TAG, "This usually means:");
                ESP_LOGE(TAG, "  - Wrong I2C pins (check SDA/SCL)");
                ESP_LOGE(TAG, "  - Camera module not powered");
                ESP_LOGE(TAG, "  - Faulty camera module");
                ESP_LOGE(TAG, "  - Wrong camera model (expected OV2640)");
                break;
            default:
                ESP_LOGE(TAG, "Unknown camera error.");
                break;
        }
        return err;
    }
    
    ESP_LOGI(TAG, "Camera initialized successfully!");
    
    // Get sensor information
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        ESP_LOGI(TAG, "========================================");
        ESP_LOGI(TAG, "Camera Sensor Information:");
        ESP_LOGI(TAG, "  PID: 0x%X", s->id.PID);
        ESP_LOGI(TAG, "  VER: 0x%X", s->id.VER);
        ESP_LOGI(TAG, "  MIDL: 0x%X", s->id.MIDL);
        ESP_LOGI(TAG, "  MIDH: 0x%X", s->id.MIDH);
        
        // Try to identify the camera model based on PID
        camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);
        if (info != NULL) {
            ESP_LOGI(TAG, "  Model: %s", get_camera_model_name(info->model));
            ESP_LOGI(TAG, "  Max Resolution: %s", get_frame_size_name(info->max_size));
            ESP_LOGI(TAG, "  JPEG Support: %s", info->support_jpeg ? "YES" : "NO");
        }
        ESP_LOGI(TAG, "========================================");
    }
    
    return ESP_OK;
}

void app_main(void) {
    // Disable brownout detector (power supply issue workaround)
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    ESP_LOGW(TAG, "Brownout detector disabled - ensure stable 5V power supply!");
    
    // Initialize NVS (required for WiFi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Print system information
    print_system_info();
    
    // Print camera configuration
    print_camera_config();
    
    // Initialize camera
    ESP_LOGI(TAG, "Initializing camera...");
    esp_err_t err = init_camera();
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "========================================");
        ESP_LOGE(TAG, "CAMERA INITIALIZATION FAILED!");
        ESP_LOGE(TAG, "========================================");
        while(1) {
            vTaskDelay(pdMS_TO_TICKS(10000));
        }
    }
    
    ESP_LOGI(TAG, "Camera initialized successfully!");
    
    // Keep camera powered on for fast response to triggers
    // (Camera sensor consumes minimal power in idle state)
    ESP_LOGI(TAG, "Camera ready - staying powered on for fast triggers");
    
    // Initialize WiFi (but don't connect yet)
    init_wifi();
    
    // Initialize trigger GPIO
    if (init_trigger_gpio() != ESP_OK) {
        ESP_LOGE(TAG, "Trigger GPIO init failed");
        while(1) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "SYSTEM READY");
    ESP_LOGI(TAG, "Waiting for trigger on GPIO%d", TRIGGER_PIN);
    ESP_LOGI(TAG, "Trigger: 100ms HIGH pulse from STM32 PB12");
    ESP_LOGI(TAG, "========================================");
    
    // Main loop: Wait for triggers
    while (1) {
        if (trigger_detected) {
            trigger_detected = false;
            process_trigger();
        }
        
        vTaskDelay(pdMS_TO_TICKS(100));  // Low power polling
    }
}
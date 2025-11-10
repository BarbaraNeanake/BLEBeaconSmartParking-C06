#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "sensor.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/i2c.h"
#include "driver/spi_slave.h"

static const char *TAG = "ESP32CAM_SPI";

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

// SPI Pins (as per blueprint.md)
#define SPI_MOSI_PIN    13  // GPIO13 -> STM32 PB15
#define SPI_MISO_PIN    14  // GPIO14 -> STM32 PB14
#define SPI_SCLK_PIN    12  // GPIO12 -> STM32 PB13
#define SPI_CS_PIN      15  // GPIO15 -> STM32 PB12 (Trigger/NSS)

// SPI Configuration
#define SPI_MAX_TRANSFER_SIZE   4096
#define DMA_CHAN                2

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
    
    .pixel_format   = PIXFORMAT_GRAYSCALE,  // Grayscale as per STM32 protocol
    .frame_size     = FRAMESIZE_QVGA,       // QVGA (320x240) = 76,800 bytes
    .jpeg_quality   = 12,                   // Not used for grayscale
    .fb_count       = 1,                    // Number of frame buffers
    .fb_location    = CAMERA_FB_IN_DRAM,    // Use DRAM since PSRAM not detected
    .grab_mode      = CAMERA_GRAB_WHEN_EMPTY
};

// Global variables
static camera_fb_t *current_fb = NULL;
static volatile bool capture_triggered = false;
static volatile bool data_ready = false;

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
    ESP_LOGI(TAG, "ESP32-CAM SPI Slave - Free heap: %lu bytes", esp_get_free_heap_size());
}

void print_camera_config(void) {
    ESP_LOGI(TAG, "Camera: %s %s", 
             get_pixel_format_name(camera_config.pixel_format),
             get_frame_size_name(camera_config.frame_size));
}

void prepare_camera_pins(void) {
    if (CAM_PIN_PWDN != -1) {
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << CAM_PIN_PWDN),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE
        };
        gpio_config(&io_conf);
        gpio_set_level(CAM_PIN_PWDN, 1);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(CAM_PIN_PWDN, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void scan_i2c_bus(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CAM_PIN_SIOD,
        .scl_io_num = CAM_PIN_SIOC,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 100000,
    };
    
    if (i2c_param_config(I2C_NUM_0, &conf) != ESP_OK) return;
    if (i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0) != ESP_OK) return;
    
    // Quick scan for OV2640 at 0x30
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (0x30 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "I2C: Camera detected at 0x30");
    }
    
    i2c_driver_delete(I2C_NUM_0);
}

esp_err_t init_ledc_for_camera(void) {
    ledc_timer_config_t ledc_timer = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = LEDC_TIMER_1_BIT,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = 20000000,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    
    esp_err_t err = ledc_timer_config(&ledc_timer);
    if (err != ESP_OK) return err;
    
    ledc_channel_config_t ledc_channel = {
        .gpio_num       = CAM_PIN_XCLK,
        .speed_mode     = LEDC_LOW_SPEED_MODE,
        .channel        = LEDC_CHANNEL_0,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = LEDC_TIMER_0,
        .duty           = 1,
        .hpoint         = 0
    };
    
    return ledc_channel_config(&ledc_channel);
}

// GPIO interrupt handler for CS pin (trigger detection)
static void IRAM_ATTR cs_interrupt_handler(void* arg) {
    // CS pin went LOW - trigger capture
    if (gpio_get_level(SPI_CS_PIN) == 0) {
        capture_triggered = true;
    }
}

esp_err_t init_cs_trigger(void) {
    // Configure CS pin as input with interrupt
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << SPI_CS_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE  // Trigger on falling edge (CS LOW)
    };
    
    esp_err_t ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CS GPIO config failed");
        return ret;
    }
    
    // Install ISR service and add handler
    gpio_install_isr_service(0);
    gpio_isr_handler_add(SPI_CS_PIN, cs_interrupt_handler, NULL);
    
    ESP_LOGI(TAG, "CS trigger ready on GPIO%d", SPI_CS_PIN);
    return ESP_OK;
}

esp_err_t init_spi_slave(void) {
    spi_bus_config_t buscfg = {
        .mosi_io_num = SPI_MOSI_PIN,
        .miso_io_num = SPI_MISO_PIN,
        .sclk_io_num = SPI_SCLK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = SPI_MAX_TRANSFER_SIZE,
    };
    
    spi_slave_interface_config_t slvcfg = {
        .mode = 0,
        .spics_io_num = -1,  // Don't use hardware CS, we handle it manually
        .queue_size = 3,
        .flags = 0,
        .post_setup_cb = NULL,
        .post_trans_cb = NULL
    };
    
    gpio_set_pull_mode(SPI_MOSI_PIN, GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(SPI_SCLK_PIN, GPIO_PULLUP_ONLY);
    
    esp_err_t ret = spi_slave_initialize(HSPI_HOST, &buscfg, &slvcfg, DMA_CHAN);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPI init failed");
        return ret;
    }
    
    ESP_LOGI(TAG, "SPI slave ready");
    return ESP_OK;
}

void capture_task(void *pvParameters) {
    while (1) {
        // Wait for CS trigger
        if (capture_triggered) {
            capture_triggered = false;
            
            ESP_LOGI(TAG, "Trigger detected");
            
            // Release old frame buffer if exists
            if (current_fb) {
                esp_camera_fb_return(current_fb);
                current_fb = NULL;
            }
            
            // Capture new image
            current_fb = esp_camera_fb_get();
            
            if (current_fb) {
                ESP_LOGI(TAG, "Captured: %d bytes (%dx%d)", 
                         current_fb->len, current_fb->width, current_fb->height);
                data_ready = true;
            } else {
                ESP_LOGE(TAG, "Capture failed");
                data_ready = false;
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void spi_transfer_task(void *pvParameters) {
    WORD_ALIGNED_ATTR uint8_t header[4];
    
    while (1) {
        // Wait for data to be ready
        if (data_ready && current_fb != NULL) {
            // Wait for STM32 to pull CS low to request data
            while (gpio_get_level(SPI_CS_PIN) == 1) {
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            
            ESP_LOGI(TAG, "Sending header");
            
            // Prepare 4-byte size header (big-endian)
            uint32_t size = current_fb->len;
            header[0] = (size >> 24) & 0xFF;
            header[1] = (size >> 16) & 0xFF;
            header[2] = (size >> 8) & 0xFF;
            header[3] = size & 0xFF;
            
            // Send header
            spi_slave_transaction_t trans_header;
            memset(&trans_header, 0, sizeof(trans_header));
            trans_header.length = 4 * 8;  // 4 bytes in bits
            trans_header.tx_buffer = header;
            
            spi_slave_queue_trans(HSPI_HOST, &trans_header, portMAX_DELAY);
            spi_slave_transaction_t *ret_trans;
            spi_slave_get_trans_result(HSPI_HOST, &ret_trans, portMAX_DELAY);
            
            // Wait for CS to go high then low again for data transfer
            while (gpio_get_level(SPI_CS_PIN) == 0) {
                vTaskDelay(pdMS_TO_TICKS(1));
            }
            
            vTaskDelay(pdMS_TO_TICKS(10));
            
            while (gpio_get_level(SPI_CS_PIN) == 1) {
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            
            ESP_LOGI(TAG, "Sending image data");
            
            // Send image data in 512-byte chunks
            uint32_t offset = 0;
            uint32_t chunk_size = 512;
            
            while (offset < current_fb->len) {
                uint32_t bytes_to_send = (offset + chunk_size > current_fb->len) ? 
                                         (current_fb->len - offset) : chunk_size;
                
                spi_slave_transaction_t trans_data;
                memset(&trans_data, 0, sizeof(trans_data));
                trans_data.length = bytes_to_send * 8;
                trans_data.tx_buffer = current_fb->buf + offset;
                
                if (spi_slave_queue_trans(HSPI_HOST, &trans_data, portMAX_DELAY) == ESP_OK) {
                    spi_slave_get_trans_result(HSPI_HOST, &ret_trans, portMAX_DELAY);
                    offset += bytes_to_send;
                } else {
                    ESP_LOGE(TAG, "SPI transfer failed at offset %lu", offset);
                    break;
                }
                
                vTaskDelay(pdMS_TO_TICKS(5));
            }
            
            ESP_LOGI(TAG, "Transfer complete");
            
            // Release frame buffer
            esp_camera_fb_return(current_fb);
            current_fb = NULL;
            data_ready = false;
        }
        
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

esp_err_t init_camera(void) {
    prepare_camera_pins();
    scan_i2c_bus();
    
    esp_err_t err = init_ledc_for_camera();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LEDC init failed");
    }
    
    err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return err;
    }
    
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);
        if (info != NULL) {
            ESP_LOGI(TAG, "Camera: %s", get_camera_model_name(info->model));
        }
    }
    
    return ESP_OK;
}

void app_main(void) {
    print_system_info();
    print_camera_config();
    
    if (init_camera() != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed - check wiring");
        while(1) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    
    if (init_cs_trigger() != ESP_OK) {
        ESP_LOGE(TAG, "CS trigger init failed");
        while(1) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    
    if (init_spi_slave() != ESP_OK) {
        ESP_LOGE(TAG, "SPI init failed");
        while(1) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    
    xTaskCreate(capture_task, "capture", 4096, NULL, 5, NULL);
    xTaskCreate(spi_transfer_task, "spi_tx", 4096, NULL, 4, NULL);
    
    ESP_LOGI(TAG, "Ready - waiting for CS trigger (GPIO%d)", SPI_CS_PIN);
    
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        ESP_LOGI(TAG, "Heap: %lu", esp_get_free_heap_size());
    }
}

/**
  ******************************************************************************
  * @file           : vl53l0x.h
  * @brief          : Header file for VL53L0X laser ranging sensor driver
  * @author         : STM32 Application
  * @date           : 2025
  ******************************************************************************
  * @attention
  *
  * VL53L0X Time-of-Flight Ranging Sensor Driver
  * Compatible with STM32F103C8T6
  *
  * Connections:
  * - SCL: PB6 (I2C1_SCL)
  * - SDA: PB7 (I2C1_SDA)
  * - GPIO1: PA0 (Interrupt pin, optional)
  * - VCC: 2.6V to 3.5V (typically 3.3V)
  * - GND: Ground
  *
  ******************************************************************************
  */

#ifndef VL53L0X_H
#define VL53L0X_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"
#include <stdint.h>
#include <stdbool.h>

/* Exported types ------------------------------------------------------------*/

/**
 * @brief VL53L0X device modes
 */
typedef enum {
    VL53L0X_MODE_SINGLE = 0x00,      // Single shot mode
    VL53L0X_MODE_CONTINUOUS = 0x02,  // Continuous ranging mode
    VL53L0X_MODE_TIMED = 0x04        // Timed ranging mode
} VL53L0X_Mode_t;

/**
 * @brief VL53L0X ranging profiles
 */
typedef enum {
    VL53L0X_PROFILE_DEFAULT = 0,     // Default ranging profile
    VL53L0X_PROFILE_HIGH_ACCURACY,   // High accuracy mode (slower)
    VL53L0X_PROFILE_LONG_RANGE,      // Long range mode (up to 2m)
    VL53L0X_PROFILE_HIGH_SPEED       // High speed mode (faster updates)
} VL53L0X_Profile_t;

/**
 * @brief VL53L0X measurement data structure
 */
typedef struct {
    uint16_t range_mm;               // Range measurement in millimeters
    uint8_t range_status;            // Range status (0 = valid)
    uint8_t signal_rate;             // Signal rate
    uint8_t ambient_rate;            // Ambient rate
    uint16_t effective_spad_count;   // Effective SPAD count
} VL53L0X_RangingData_t;

/**
 * @brief VL53L0X device information
 */
typedef struct {
    uint8_t model_id;                // Model ID (should be 0xEE)
    uint8_t module_type;             // Module type
    uint8_t revision_id;             // Revision ID
    uint16_t product_id;             // Product ID
} VL53L0X_DeviceInfo_t;

/**
 * @brief VL53L0X status codes
 */
typedef enum {
    VL53L0X_OK = 0,                  // Operation successful
    VL53L0X_ERROR = 1,               // General error
    VL53L0X_ERROR_I2C = 2,           // I2C communication error
    VL53L0X_ERROR_TIMEOUT = 3,       // Timeout error
    VL53L0X_ERROR_INVALID_PARAMS = 4 // Invalid parameters
} VL53L0X_Status_t;

/* Exported constants --------------------------------------------------------*/

/** @defgroup VL53L0X_I2C_Address
  * @{
  */
#define VL53L0X_I2C_ADDR                    0x29  // 7-bit address
#define VL53L0X_I2C_ADDR_8BIT               (VL53L0X_I2C_ADDR << 1)
/**
  * @}
  */

/** @defgroup VL53L0X_Register_Addresses
  * @{
  */
#define VL53L0X_REG_SYSRANGE_START                    0x00
#define VL53L0X_REG_SYSTEM_THRESH_HIGH                0x0C
#define VL53L0X_REG_SYSTEM_THRESH_LOW                 0x0E
#define VL53L0X_REG_SYSTEM_SEQUENCE_CONFIG            0x01
#define VL53L0X_REG_SYSTEM_RANGE_CONFIG               0x09
#define VL53L0X_REG_SYSTEM_INTERMEASUREMENT_PERIOD    0x04
#define VL53L0X_REG_SYSTEM_INTERRUPT_CONFIG_GPIO      0x0A
#define VL53L0X_REG_GPIO_HV_MUX_ACTIVE_HIGH           0x84
#define VL53L0X_REG_SYSTEM_INTERRUPT_CLEAR            0x0B
#define VL53L0X_REG_RESULT_INTERRUPT_STATUS           0x13
#define VL53L0X_REG_RESULT_RANGE_STATUS               0x14
#define VL53L0X_REG_RESULT_CORE_AMBIENT_WINDOW_EVENTS 0xBC
#define VL53L0X_REG_RESULT_CORE_RANGING_TOTAL_EVENTS  0xC0
#define VL53L0X_REG_RESULT_CORE_AMBIENT_WINDOW_EVENTS 0xBC
#define VL53L0X_REG_RESULT_PEAK_SIGNAL_RATE_REF       0xB6
#define VL53L0X_REG_ALGO_PART_TO_PART_RANGE_OFFSET_MM 0x28
#define VL53L0X_REG_I2C_SLAVE_DEVICE_ADDRESS          0x8A
#define VL53L0X_REG_MSRC_CONFIG_CONTROL               0x60
#define VL53L0X_REG_PRE_RANGE_CONFIG_MIN_SNR          0x27
#define VL53L0X_REG_PRE_RANGE_CONFIG_VALID_PHASE_LOW  0x56
#define VL53L0X_REG_PRE_RANGE_CONFIG_VALID_PHASE_HIGH 0x57
#define VL53L0X_REG_PRE_RANGE_MIN_COUNT_RATE_RTN_LIMIT 0x64
#define VL53L0X_REG_FINAL_RANGE_CONFIG_MIN_SNR        0x67
#define VL53L0X_REG_FINAL_RANGE_CONFIG_VALID_PHASE_LOW 0x47
#define VL53L0X_REG_FINAL_RANGE_CONFIG_VALID_PHASE_HIGH 0x48
#define VL53L0X_REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT 0x44
#define VL53L0X_REG_PRE_RANGE_CONFIG_SIGMA_THRESH_HI  0x61
#define VL53L0X_REG_PRE_RANGE_CONFIG_SIGMA_THRESH_LO  0x62
#define VL53L0X_REG_PRE_RANGE_CONFIG_VCSEL_PERIOD     0x50
#define VL53L0X_REG_PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI 0x51
#define VL53L0X_REG_PRE_RANGE_CONFIG_TIMEOUT_MACROP_LO 0x52
#define VL53L0X_REG_SYSTEM_HISTOGRAM_BIN              0x81
#define VL53L0X_REG_HISTOGRAM_CONFIG_INITIAL_PHASE_SELECT 0x33
#define VL53L0X_REG_HISTOGRAM_CONFIG_READOUT_CTRL     0x55
#define VL53L0X_REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD   0x70
#define VL53L0X_REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI 0x71
#define VL53L0X_REG_FINAL_RANGE_CONFIG_TIMEOUT_MACROP_LO 0x72
#define VL53L0X_REG_CROSSTALK_COMPENSATION_PEAK_RATE_MCPS 0x20
#define VL53L0X_REG_MSRC_CONFIG_TIMEOUT_MACROP        0x46
#define VL53L0X_REG_SOFT_RESET_GO2_SOFT_RESET_N       0xBF
#define VL53L0X_REG_IDENTIFICATION_MODEL_ID           0xC0
#define VL53L0X_REG_IDENTIFICATION_REVISION_ID        0xC2
#define VL53L0X_REG_OSC_CALIBRATE_VAL                 0xF8
#define VL53L0X_REG_GLOBAL_CONFIG_VCSEL_WIDTH         0x32
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_0  0xB0
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_1  0xB1
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_2  0xB2
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_3  0xB3
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_4  0xB4
#define VL53L0X_REG_GLOBAL_CONFIG_SPAD_ENABLES_REF_5  0xB5
#define VL53L0X_REG_GLOBAL_CONFIG_REF_EN_START_SELECT 0xB6
#define VL53L0X_REG_DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD 0x4E
#define VL53L0X_REG_DYNAMIC_SPAD_REF_EN_START_OFFSET  0x4F
#define VL53L0X_REG_POWER_MANAGEMENT_GO1_POWER_FORCE  0x80
#define VL53L0X_REG_VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV 0x89
#define VL53L0X_REG_ALGO_PHASECAL_LIM                 0x30
#define VL53L0X_REG_ALGO_PHASECAL_CONFIG_TIMEOUT      0x30
/**
  * @}
  */

/** @defgroup VL53L0X_Device_Parameters
  * @{
  */
#define VL53L0X_EXPECTED_DEVICE_ID          0xEE
#define VL53L0X_DEFAULT_MAX_LOOP            200
#define VL53L0X_MAX_STRING_LENGTH           32
#define VL53L0X_MEASUREMENT_TIMEOUT_MS      500
/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
#define VL53L0X_TIMEOUT_MS(start)   ((HAL_GetTick() - (start)) > VL53L0X_MEASUREMENT_TIMEOUT_MS)

/* Exported functions prototypes ---------------------------------------------*/

/** @defgroup VL53L0X_Initialization_Functions Initialization Functions
  * @{
  */
VL53L0X_Status_t VL53L0X_Init(I2C_HandleTypeDef *hi2c);
VL53L0X_Status_t VL53L0X_DeInit(I2C_HandleTypeDef *hi2c);
VL53L0X_Status_t VL53L0X_Reset(I2C_HandleTypeDef *hi2c);
VL53L0X_Status_t VL53L0X_GetDeviceInfo(I2C_HandleTypeDef *hi2c, VL53L0X_DeviceInfo_t *info);
bool VL53L0X_IsDeviceReady(I2C_HandleTypeDef *hi2c);
/**
  * @}
  */

/** @defgroup VL53L0X_Configuration_Functions Configuration Functions
  * @{
  */
VL53L0X_Status_t VL53L0X_SetDeviceAddress(I2C_HandleTypeDef *hi2c, uint8_t new_addr);
VL53L0X_Status_t VL53L0X_SetMeasurementTimingBudget(I2C_HandleTypeDef *hi2c, uint32_t budget_us);
VL53L0X_Status_t VL53L0X_GetMeasurementTimingBudget(I2C_HandleTypeDef *hi2c, uint32_t *budget_us);
VL53L0X_Status_t VL53L0X_SetProfile(I2C_HandleTypeDef *hi2c, VL53L0X_Profile_t profile);
VL53L0X_Status_t VL53L0X_SetOffsetCalibration(I2C_HandleTypeDef *hi2c, int8_t offset_mm);
VL53L0X_Status_t VL53L0X_GetOffsetCalibration(I2C_HandleTypeDef *hi2c, int8_t *offset_mm);
VL53L0X_Status_t VL53L0X_SetXTalkCompensation(I2C_HandleTypeDef *hi2c, uint16_t xtalk_mcps);
/**
  * @}
  */

/** @defgroup VL53L0X_Ranging_Functions Ranging Functions
  * @{
  */
VL53L0X_Status_t VL53L0X_StartMeasurement(I2C_HandleTypeDef *hi2c, VL53L0X_Mode_t mode);
VL53L0X_Status_t VL53L0X_StopMeasurement(I2C_HandleTypeDef *hi2c);
VL53L0X_Status_t VL53L0X_ReadRangeData(I2C_HandleTypeDef *hi2c, VL53L0X_RangingData_t *data);
VL53L0X_Status_t VL53L0X_ReadRangeSingleMillimeters(I2C_HandleTypeDef *hi2c, uint16_t *distance_mm);
VL53L0X_Status_t VL53L0X_ReadRangeContinuousMillimeters(I2C_HandleTypeDef *hi2c, uint16_t *distance_mm);
bool VL53L0X_IsDataReady(I2C_HandleTypeDef *hi2c);
VL53L0X_Status_t VL53L0X_ClearInterrupt(I2C_HandleTypeDef *hi2c);
/**
  * @}
  */

/** @defgroup VL53L0X_Interrupt_Functions Interrupt Functions
  * @{
  */
VL53L0X_Status_t VL53L0X_ConfigureInterrupt(I2C_HandleTypeDef *hi2c, bool enable);
VL53L0X_Status_t VL53L0X_SetInterruptThresholds(I2C_HandleTypeDef *hi2c, uint16_t low_mm, uint16_t high_mm);
/**
  * @}
  */

/** @defgroup VL53L0X_Utility_Functions Utility Functions
  * @{
  */
uint16_t VL53L0X_ConvertMmToCm(uint16_t distance_mm);
float VL53L0X_ConvertMmToMeters(uint16_t distance_mm);
uint16_t VL53L0X_ConvertMmToInches(uint16_t distance_mm);
const char* VL53L0X_GetStatusString(VL53L0X_Status_t status);
const char* VL53L0X_GetRangeStatusString(uint8_t range_status);
/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* VL53L0X_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

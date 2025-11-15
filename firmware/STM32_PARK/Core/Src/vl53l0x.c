/**
  ******************************************************************************
  * @file           : vl53l0x.c
  * @brief          : VL53L0X laser ranging sensor driver implementation (SIMPLIFIED)
  * @author         : STM32 Application
  * @date           : 2025
  ******************************************************************************
  */

#include <string.h>
#include <vl53l0x.h>

/* Private variables ---------------------------------------------------------*/
static uint8_t stop_variable = 0;
static uint32_t measurement_timing_budget_us = 0;

/* Private function prototypes -----------------------------------------------*/
static HAL_StatusTypeDef VL53L0X_WriteReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t data);
static HAL_StatusTypeDef VL53L0X_WriteReg16(I2C_HandleTypeDef *hi2c, uint8_t reg, uint16_t data);
static HAL_StatusTypeDef VL53L0X_WriteReg32(I2C_HandleTypeDef *hi2c, uint8_t reg, uint32_t data);
static HAL_StatusTypeDef VL53L0X_ReadReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *data);
static HAL_StatusTypeDef VL53L0X_ReadReg16(I2C_HandleTypeDef *hi2c, uint8_t reg, uint16_t *data);
static HAL_StatusTypeDef VL53L0X_ReadMulti(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *data, uint8_t len);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Write single byte to register
  */
static HAL_StatusTypeDef VL53L0X_WriteReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t data)
{
    uint8_t buf[2] = {reg, data};
    return HAL_I2C_Master_Transmit(hi2c, VL53L0X_I2C_ADDR_8BIT, buf, 2, 1000);
}

/**
  * @brief  Write 16-bit value to register
  */
static HAL_StatusTypeDef VL53L0X_WriteReg16(I2C_HandleTypeDef *hi2c, uint8_t reg, uint16_t data)
{
    uint8_t buf[3] = {reg, (uint8_t)(data >> 8), (uint8_t)(data & 0xFF)};
    return HAL_I2C_Master_Transmit(hi2c, VL53L0X_I2C_ADDR_8BIT, buf, 3, 1000);
}

/**
  * @brief  Write 32-bit value to register
  */
static HAL_StatusTypeDef VL53L0X_WriteReg32(I2C_HandleTypeDef *hi2c, uint8_t reg, uint32_t data)
{
    uint8_t buf[5] = {reg, (uint8_t)(data >> 24), (uint8_t)(data >> 16),
                      (uint8_t)(data >> 8), (uint8_t)(data & 0xFF)};
    return HAL_I2C_Master_Transmit(hi2c, VL53L0X_I2C_ADDR_8BIT, buf, 5, 1000);
}

/**
  * @brief  Read single byte from register
  */
static HAL_StatusTypeDef VL53L0X_ReadReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *data)
{
    return HAL_I2C_Mem_Read(hi2c, VL53L0X_I2C_ADDR_8BIT, reg, I2C_MEMADD_SIZE_8BIT, data, 1, 1000);
}

/**
  * @brief  Read 16-bit value from register
  */
static HAL_StatusTypeDef VL53L0X_ReadReg16(I2C_HandleTypeDef *hi2c, uint8_t reg, uint16_t *data)
{
    uint8_t buf[2];
    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(hi2c, VL53L0X_I2C_ADDR_8BIT, reg, I2C_MEMADD_SIZE_8BIT, buf, 2, 1000);
    *data = (buf[0] << 8) | buf[1];
    return status;
}

/**
  * @brief  Read multiple bytes from register
  */
static HAL_StatusTypeDef VL53L0X_ReadMulti(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *data, uint8_t len)
{
    return HAL_I2C_Mem_Read(hi2c, VL53L0X_I2C_ADDR_8BIT, reg, I2C_MEMADD_SIZE_8BIT, data, len, 1000);
}

/**
  * @brief  Check if device is ready
  */
bool VL53L0X_IsDeviceReady(I2C_HandleTypeDef *hi2c)
{
    return (HAL_I2C_IsDeviceReady(hi2c, VL53L0X_I2C_ADDR_8BIT, 3, 100) == HAL_OK);
}

/**
  * @brief  Initialize VL53L0X sensor - SIMPLIFIED VERSION
  */
VL53L0X_Status_t VL53L0X_Init(I2C_HandleTypeDef *hi2c)
{
    uint8_t val;

    // Check device ID
    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_IDENTIFICATION_MODEL_ID, &val) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    if (val != VL53L0X_EXPECTED_DEVICE_ID) {
        return VL53L0X_ERROR;
    }

    // Minimal initialization sequence - based on Pololu library
    // Set I2C standard mode
    VL53L0X_WriteReg(hi2c, 0x88, 0x00);

    VL53L0X_WriteReg(hi2c, 0x80, 0x01);
    VL53L0X_WriteReg(hi2c, 0xFF, 0x01);
    VL53L0X_WriteReg(hi2c, 0x00, 0x00);
    VL53L0X_ReadReg(hi2c, 0x91, &stop_variable);
    VL53L0X_WriteReg(hi2c, 0x00, 0x01);
    VL53L0X_WriteReg(hi2c, 0xFF, 0x00);
    VL53L0X_WriteReg(hi2c, 0x80, 0x00);

    // Disable SIGNAL_RATE_MSRC and SIGNAL_RATE_PRE_RANGE limit checks
    VL53L0X_WriteReg(hi2c, VL53L0X_REG_MSRC_CONFIG_CONTROL, 0x12);

    // Set final range signal rate limit to 0.25 MCPS
    VL53L0X_WriteReg16(hi2c, VL53L0X_REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT, 32);

    VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSTEM_SEQUENCE_CONFIG, 0xFF);

    // Set default timing budget
    measurement_timing_budget_us = 33000;

    return VL53L0X_OK;
}

/**
  * @brief  De-initialize VL53L0X sensor
  */
VL53L0X_Status_t VL53L0X_DeInit(I2C_HandleTypeDef *hi2c)
{
    return VL53L0X_StopMeasurement(hi2c);
}

/**
  * @brief  Reset VL53L0X sensor
  */
VL53L0X_Status_t VL53L0X_Reset(I2C_HandleTypeDef *hi2c)
{
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_SOFT_RESET_GO2_SOFT_RESET_N, 0x00) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    HAL_Delay(10);
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_SOFT_RESET_GO2_SOFT_RESET_N, 0x01) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    HAL_Delay(10);
    return VL53L0X_OK;
}

/**
  * @brief  Get device information
  */
VL53L0X_Status_t VL53L0X_GetDeviceInfo(I2C_HandleTypeDef *hi2c, VL53L0X_DeviceInfo_t *info)
{
    uint8_t val;

    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_IDENTIFICATION_MODEL_ID, &info->model_id) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_IDENTIFICATION_MODEL_ID + 1, &info->module_type) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_IDENTIFICATION_REVISION_ID, &info->revision_id) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    uint16_t prod_id;
    if (VL53L0X_ReadReg16(hi2c, VL53L0X_REG_IDENTIFICATION_MODEL_ID + 2, &prod_id) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    info->product_id = prod_id;

    return VL53L0X_OK;
}

/**
  * @brief  Set device I2C address
  */
VL53L0X_Status_t VL53L0X_SetDeviceAddress(I2C_HandleTypeDef *hi2c, uint8_t new_addr)
{
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_I2C_SLAVE_DEVICE_ADDRESS, new_addr & 0x7F) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Set measurement profile
  */
VL53L0X_Status_t VL53L0X_SetProfile(I2C_HandleTypeDef *hi2c, VL53L0X_Profile_t profile)
{
    switch(profile) {
        case VL53L0X_PROFILE_HIGH_ACCURACY:
            VL53L0X_SetMeasurementTimingBudget(hi2c, 200000);
            break;
        case VL53L0X_PROFILE_LONG_RANGE:
            VL53L0X_SetMeasurementTimingBudget(hi2c, 33000);
            // Enable long range
            VL53L0X_WriteReg16(hi2c, VL53L0X_REG_FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT, 13);
            VL53L0X_WriteReg(hi2c, VL53L0X_REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD, 0x0E);
            break;
        case VL53L0X_PROFILE_HIGH_SPEED:
            VL53L0X_SetMeasurementTimingBudget(hi2c, 20000);
            break;
        default:
            VL53L0X_SetMeasurementTimingBudget(hi2c, 33000);
            break;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Set measurement timing budget
  */
VL53L0X_Status_t VL53L0X_SetMeasurementTimingBudget(I2C_HandleTypeDef *hi2c, uint32_t budget_us)
{
    measurement_timing_budget_us = budget_us;
    return VL53L0X_OK;
}

/**
  * @brief  Get measurement timing budget
  */
VL53L0X_Status_t VL53L0X_GetMeasurementTimingBudget(I2C_HandleTypeDef *hi2c, uint32_t *budget_us)
{
    *budget_us = measurement_timing_budget_us;
    return VL53L0X_OK;
}

/**
  * @brief  Set offset calibration
  */
VL53L0X_Status_t VL53L0X_SetOffsetCalibration(I2C_HandleTypeDef *hi2c, int8_t offset_mm)
{
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_ALGO_PART_TO_PART_RANGE_OFFSET_MM, offset_mm) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Get offset calibration
  */
VL53L0X_Status_t VL53L0X_GetOffsetCalibration(I2C_HandleTypeDef *hi2c, int8_t *offset_mm)
{
    uint8_t val;
    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_ALGO_PART_TO_PART_RANGE_OFFSET_MM, &val) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    *offset_mm = (int8_t)val;
    return VL53L0X_OK;
}

/**
  * @brief  Set crosstalk compensation
  */
VL53L0X_Status_t VL53L0X_SetXTalkCompensation(I2C_HandleTypeDef *hi2c, uint16_t xtalk_mcps)
{
    if (VL53L0X_WriteReg16(hi2c, VL53L0X_REG_CROSSTALK_COMPENSATION_PEAK_RATE_MCPS, xtalk_mcps) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Start measurement
  */
VL53L0X_Status_t VL53L0X_StartMeasurement(I2C_HandleTypeDef *hi2c, VL53L0X_Mode_t mode)
{
    VL53L0X_WriteReg(hi2c, 0x80, 0x01);
    VL53L0X_WriteReg(hi2c, 0xFF, 0x01);
    VL53L0X_WriteReg(hi2c, 0x00, 0x00);
    VL53L0X_WriteReg(hi2c, 0x91, stop_variable);
    VL53L0X_WriteReg(hi2c, 0x00, 0x01);
    VL53L0X_WriteReg(hi2c, 0xFF, 0x00);
    VL53L0X_WriteReg(hi2c, 0x80, 0x00);

    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSRANGE_START, mode) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Stop measurement
  */
VL53L0X_Status_t VL53L0X_StopMeasurement(I2C_HandleTypeDef *hi2c)
{
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSRANGE_START, 0x01) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    VL53L0X_WriteReg(hi2c, 0xFF, 0x01);
    VL53L0X_WriteReg(hi2c, 0x00, 0x00);
    VL53L0X_WriteReg(hi2c, 0x91, 0x00);
    VL53L0X_WriteReg(hi2c, 0x00, 0x01);
    VL53L0X_WriteReg(hi2c, 0xFF, 0x00);

    return VL53L0X_OK;
}

/**
  * @brief  Check if data is ready
  */
bool VL53L0X_IsDataReady(I2C_HandleTypeDef *hi2c)
{
    uint8_t val;
    if (VL53L0X_ReadReg(hi2c, VL53L0X_REG_RESULT_INTERRUPT_STATUS, &val) != HAL_OK) {
        return false;
    }
    return (val & 0x07) != 0;
}

/**
  * @brief  Read complete ranging data
  */
VL53L0X_Status_t VL53L0X_ReadRangeData(I2C_HandleTypeDef *hi2c, VL53L0X_RangingData_t *data)
{
    uint8_t range_status_buffer[12];

    if (VL53L0X_ReadMulti(hi2c, VL53L0X_REG_RESULT_RANGE_STATUS, range_status_buffer, 12) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }

    // According to Pololu VL53L0X library and datasheet:
    // Byte 0 bits [6:3] contain DeviceRangeStatus (0-13)
    // We need to map this to simpler error codes
    uint8_t device_range_status = (range_status_buffer[0] >> 3) & 0x0F;

    // Map device range status to simple range status (0-7)
    // Based on ST VL53L0X API v1.0
    if (device_range_status == 11) {
        data->range_status = 0;  // Valid, no error
    } else if (device_range_status == 9 || device_range_status == 6) {
        data->range_status = 1;  // Sigma fail
    } else if (device_range_status == 4 || device_range_status == 5) {
        data->range_status = 2;  // Signal fail
    } else if (device_range_status == 8) {
        data->range_status = 3;  // Min range fail (target too close)
    } else if (device_range_status == 3 || device_range_status == 10) {
        data->range_status = 4;  // Phase fail
    } else if (device_range_status == 1 || device_range_status == 2) {
        data->range_status = 5;  // Hardware fail
    } else {
        data->range_status = device_range_status;  // Other
    }

    data->signal_rate = range_status_buffer[6];
    data->ambient_rate = range_status_buffer[7];
    data->effective_spad_count = (range_status_buffer[2] << 8) | range_status_buffer[3];
    data->range_mm = (range_status_buffer[10] << 8) | range_status_buffer[11];

    return VL53L0X_OK;
}

/**
  * @brief  Read range in single-shot mode
  */
VL53L0X_Status_t VL53L0X_ReadRangeSingleMillimeters(I2C_HandleTypeDef *hi2c, uint16_t *distance_mm)
{
    VL53L0X_Status_t status;
    VL53L0X_RangingData_t data;

    status = VL53L0X_StartMeasurement(hi2c, VL53L0X_MODE_SINGLE);
    if (status != VL53L0X_OK) return status;

    uint32_t start = HAL_GetTick();
    while (!VL53L0X_IsDataReady(hi2c)) {
        if (HAL_GetTick() - start > VL53L0X_MEASUREMENT_TIMEOUT_MS) {
            return VL53L0X_ERROR_TIMEOUT;
        }
        HAL_Delay(1);
    }

    status = VL53L0X_ReadRangeData(hi2c, &data);
    if (status != VL53L0X_OK) return status;

    *distance_mm = data.range_mm;
    VL53L0X_ClearInterrupt(hi2c);

    return VL53L0X_OK;
}

/**
  * @brief  Read range in continuous mode
  */
VL53L0X_Status_t VL53L0X_ReadRangeContinuousMillimeters(I2C_HandleTypeDef *hi2c, uint16_t *distance_mm)
{
    VL53L0X_RangingData_t data;
    VL53L0X_Status_t status;

    uint32_t start = HAL_GetTick();
    while (!VL53L0X_IsDataReady(hi2c)) {
        if (HAL_GetTick() - start > VL53L0X_MEASUREMENT_TIMEOUT_MS) {
            return VL53L0X_ERROR_TIMEOUT;
        }
        HAL_Delay(1);
    }

    status = VL53L0X_ReadRangeData(hi2c, &data);
    if (status != VL53L0X_OK) return status;

    *distance_mm = data.range_mm;
    VL53L0X_ClearInterrupt(hi2c);

    return VL53L0X_OK;
}

/**
  * @brief  Clear interrupt
  */
VL53L0X_Status_t VL53L0X_ClearInterrupt(I2C_HandleTypeDef *hi2c)
{
    if (VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSTEM_INTERRUPT_CLEAR, 0x01) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Configure interrupt
  */
VL53L0X_Status_t VL53L0X_ConfigureInterrupt(I2C_HandleTypeDef *hi2c, bool enable)
{
    if (enable) {
        VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04);
        uint8_t val;
        VL53L0X_ReadReg(hi2c, VL53L0X_REG_GPIO_HV_MUX_ACTIVE_HIGH, &val);
        VL53L0X_WriteReg(hi2c, VL53L0X_REG_GPIO_HV_MUX_ACTIVE_HIGH, val & ~0x10);
    } else {
        VL53L0X_WriteReg(hi2c, VL53L0X_REG_SYSTEM_INTERRUPT_CONFIG_GPIO, 0x00);
    }
    return VL53L0X_OK;
}

/**
  * @brief  Set interrupt thresholds
  */
VL53L0X_Status_t VL53L0X_SetInterruptThresholds(I2C_HandleTypeDef *hi2c, uint16_t low_mm, uint16_t high_mm)
{
    if (VL53L0X_WriteReg16(hi2c, VL53L0X_REG_SYSTEM_THRESH_LOW, low_mm) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    if (VL53L0X_WriteReg16(hi2c, VL53L0X_REG_SYSTEM_THRESH_HIGH, high_mm) != HAL_OK) {
        return VL53L0X_ERROR_I2C;
    }
    return VL53L0X_OK;
}

/**
  * @brief  Convert millimeters to centimeters
  */
uint16_t VL53L0X_ConvertMmToCm(uint16_t distance_mm)
{
    return distance_mm / 10;
}

/**
  * @brief  Convert millimeters to meters
  */
float VL53L0X_ConvertMmToMeters(uint16_t distance_mm)
{
    return distance_mm / 1000.0f;
}

/**
  * @brief  Convert millimeters to inches
  */
uint16_t VL53L0X_ConvertMmToInches(uint16_t distance_mm)
{
    return (distance_mm * 10) / 254;
}

/**
  * @brief  Get status string
  */
const char* VL53L0X_GetStatusString(VL53L0X_Status_t status)
{
    switch(status) {
        case VL53L0X_OK:
            return "OK";
        case VL53L0X_ERROR:
            return "Error";
        case VL53L0X_ERROR_I2C:
            return "I2C Error";
        case VL53L0X_ERROR_TIMEOUT:
            return "Timeout";
        case VL53L0X_ERROR_INVALID_PARAMS:
            return "Invalid Parameters";
        default:
            return "Unknown Error";
    }
}

/**
  * @brief  Get range status string
  */
const char* VL53L0X_GetRangeStatusString(uint8_t range_status)
{
    switch(range_status) {
        case 0:
            return "Valid - Good Measurement";
        case 1:
            return "Sigma Fail - Measurement variance too high";
        case 2:
            return "Signal Fail - Signal too weak";
        case 3:
            return "Min Range Fail - Target too close";
        case 4:
            return "Phase Fail - Phase error";
        case 5:
            return "Hardware Fail - Hardware error";
        case 6:
            return "No Update";
        case 7:
            return "Wraparound";
        default:
            return "Unknown Status";
    }
}

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE*****/

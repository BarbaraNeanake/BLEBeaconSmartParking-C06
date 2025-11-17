#ifndef JPEG_ENCODER_H
#define JPEG_ENCODER_H

#include <stdint.h>

/**
 * @brief Simple JPEG encoder for grayscale images
 * @param input_buffer Raw grayscale image data (640x480)
 * @param input_size Size of input buffer
 * @param output_buffer Buffer to store JPEG output
 * @param output_max_size Maximum size of output buffer
 * @param quality JPEG quality (1-100, recommended 75-80)
 * @return Actual size of compressed JPEG, or 0 on error
 */
uint32_t JPEG_EncodeGrayscale(const uint8_t* input_buffer, uint32_t input_size,
                               uint8_t* output_buffer, uint32_t output_max_size,
                               uint8_t quality);

#endif // JPEG_ENCODER_H

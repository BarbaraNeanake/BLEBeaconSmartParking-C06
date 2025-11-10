#ifndef JPEG_STREAMING_H
#define JPEG_STREAMING_H

#include <stdint.h>

/**
 * @brief Streaming JPEG encoder context
 * Allows encoding images larger than RAM by processing row-by-row
 */
typedef struct {
    uint8_t* output_buffer;      // Output JPEG buffer
    uint32_t output_size;        // Current output size
    uint32_t output_max;         // Max output size
    uint16_t width;              // Image width
    uint16_t height;             // Image height
    uint16_t rows_processed;     // Rows processed so far
    uint8_t quality;             // JPEG quality
    uint8_t quant_table[64];     // Quantization table
} JPEGStreamContext;

/**
 * @brief Initialize streaming JPEG encoder
 * @param ctx Encoder context
 * @param output Output buffer for JPEG data
 * @param output_max Maximum output buffer size
 * @param width Image width
 * @param height Image height
 * @param quality JPEG quality (1-100)
 * @return 1 on success, 0 on error
 */
uint8_t JPEG_StreamInit(JPEGStreamContext* ctx, uint8_t* output, uint32_t output_max,
                        uint16_t width, uint16_t height, uint8_t quality);

/**
 * @brief Process 8 rows of image data
 * @param ctx Encoder context
 * @param row_data 8 rows of grayscale pixels (width * 8 bytes)
 * @return 1 on success, 0 on error
 */
uint8_t JPEG_StreamProcessRows(JPEGStreamContext* ctx, const uint8_t* row_data);

/**
 * @brief Finalize JPEG encoding and get final size
 * @param ctx Encoder context
 * @return Final JPEG size in bytes
 */
uint32_t JPEG_StreamFinalize(JPEGStreamContext* ctx);

#endif // JPEG_STREAMING_H

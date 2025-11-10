#include "jpeg_streaming.h"
#include <string.h>

// Write helper functions
static void write_byte(uint8_t** ptr, uint8_t value) {
    *(*ptr)++ = value;
}

static void write_word(uint8_t** ptr, uint16_t value) {
    *(*ptr)++ = (value >> 8) & 0xFF;
    *(*ptr)++ = value & 0xFF;
}

// Standard luminance quantization table
static const uint8_t std_quant[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

uint8_t JPEG_StreamInit(JPEGStreamContext* ctx, uint8_t* output, uint32_t output_max,
                        uint16_t width, uint16_t height, uint8_t quality) {
    if (!ctx || !output || width == 0 || height == 0) return 0;
    
    ctx->output_buffer = output;
    ctx->output_size = 0;
    ctx->output_max = output_max;
    ctx->width = width;
    ctx->height = height;
    ctx->rows_processed = 0;
    ctx->quality = quality;
    
    // Calculate quality-scaled quantization table
    int scale = (quality < 50) ? (5000 / quality) : (200 - quality * 2);
    for (int i = 0; i < 64; i++) {
        int val = (std_quant[i] * scale + 50) / 100;
        if (val < 1) val = 1;
        if (val > 255) val = 255;
        ctx->quant_table[i] = (uint8_t)val;
    }
    
    // Write JPEG header
    uint8_t* ptr = output;
    
    // SOI
    write_word(&ptr, 0xFFD8);
    
    // APP0 (JFIF)
    write_word(&ptr, 0xFFE0);
    write_word(&ptr, 16);
    write_byte(&ptr, 'J');
    write_byte(&ptr, 'F');
    write_byte(&ptr, 'I');
    write_byte(&ptr, 'F');
    write_byte(&ptr, 0);
    write_word(&ptr, 0x0101);
    write_byte(&ptr, 0);
    write_word(&ptr, 1);
    write_word(&ptr, 1);
    write_byte(&ptr, 0);
    write_byte(&ptr, 0);
    
    // DQT
    write_word(&ptr, 0xFFDB);
    write_word(&ptr, 67);
    write_byte(&ptr, 0);
    memcpy(ptr, ctx->quant_table, 64);
    ptr += 64;
    
    // SOF0
    write_word(&ptr, 0xFFC0);
    write_word(&ptr, 11);
    write_byte(&ptr, 8);
    write_word(&ptr, height);
    write_word(&ptr, width);
    write_byte(&ptr, 1);
    write_byte(&ptr, 1);
    write_byte(&ptr, 0x11);
    write_byte(&ptr, 0);
    
    // DHT (simplified)
    write_word(&ptr, 0xFFC4);
    write_word(&ptr, 31);
    write_byte(&ptr, 0x00);
    for (int i = 0; i < 16; i++) {
        write_byte(&ptr, (i == 0) ? 12 : 0);
    }
    for (int i = 0; i < 12; i++) {
        write_byte(&ptr, i);
    }
    
    // SOS
    write_word(&ptr, 0xFFDA);
    write_word(&ptr, 8);
    write_byte(&ptr, 1);
    write_byte(&ptr, 1);
    write_byte(&ptr, 0x00);
    write_byte(&ptr, 0);
    write_byte(&ptr, 63);
    write_byte(&ptr, 0);
    
    ctx->output_size = (uint32_t)(ptr - output);
    return 1;
}

uint8_t JPEG_StreamProcessRows(JPEGStreamContext* ctx, const uint8_t* row_data) {
    if (!ctx || !row_data) return 0;
    if (ctx->rows_processed + 8 > ctx->height) return 0;
    
    uint8_t* ptr = ctx->output_buffer + ctx->output_size;
    uint32_t space = ctx->output_max - ctx->output_size;
    
    // Simple compression: subsample by 4 for each 8x8 block
    for (uint16_t x = 0; x < ctx->width; x += 4) {
        if (space < 10) return 0;
        
        uint8_t avg = row_data[x];
        
        // Write byte, escape 0xFF
        if (avg == 0xFF) {
            write_byte(&ptr, 0xFF);
            write_byte(&ptr, 0x00);
            space -= 2;
        } else {
            write_byte(&ptr, avg);
            space -= 1;
        }
    }
    
    ctx->output_size = (uint32_t)(ptr - ctx->output_buffer);
    ctx->rows_processed += 8;
    
    return 1;
}

uint32_t JPEG_StreamFinalize(JPEGStreamContext* ctx) {
    if (!ctx) return 0;
    
    uint8_t* ptr = ctx->output_buffer + ctx->output_size;
    
    // EOI
    write_word(&ptr, 0xFFD9);
    
    ctx->output_size = (uint32_t)(ptr - ctx->output_buffer);
    return ctx->output_size;
}

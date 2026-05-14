#include "kan_quant_inference.h"

#include <math.h>
#include <stdint.h>

#include "kan_model_quant.h"

static inline int kanq_layer_slot(int layer, int input_index) {
    return layer * KANQ_MAX_INPUT_DIM + input_index;
}

static inline int kanq_edge_slot(int layer, int input_index, int output_index) {
    return (layer * KANQ_MAX_INPUT_DIM + input_index) * KANQ_MAX_OUTPUT_DIM + output_index;
}

static inline float kanq_clip(float value, float lower, float upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static inline float kanq_fake_quant(float value, float amax) {
    if (amax <= 0.0f) {
        return value;
    }

    const float clipped = kanq_clip(value, -amax, amax);
    const float step = amax / (float)KANQ_QMAX;
    if (step <= 0.0f) {
        return clipped;
    }

    return roundf(clipped / step) * step;
}

static inline int kanq_lut_index(int slot, float x) {
    const float support_min = KANQ_LAYER_SUPPORT_MIN[slot];
    const float support_max = KANQ_LAYER_SUPPORT_MAX[slot];
    const float support_span = support_max - support_min;
    if (support_span <= 0.0f || KANQ_LUT_SIZE <= 1) {
        return 0;
    }

    const float clipped = kanq_clip(x, support_min, support_max);
    const float normalized = (clipped - support_min) / support_span;
    int index = (int)lroundf(normalized * (float)(KANQ_LUT_SIZE - 1));
    if (index < 0) {
        index = 0;
    }
    if (index >= KANQ_LUT_SIZE) {
        index = KANQ_LUT_SIZE - 1;
    }
    return index;
}

float kan_quant_infer_vector(const float *restrict input) {
    float current[KANQ_MAX_LAYER_WIDTH] = {0.0f};
    float next[KANQ_MAX_LAYER_WIDTH] = {0.0f};

    for (int index = 0; index < KANQ_INPUT_DIM; ++index) {
        current[index] = input[index];
    }

    for (int layer = 0; layer < KANQ_NUM_LAYERS; ++layer) {
        const int input_dim = KANQ_LAYER_INPUT_DIMS[layer];
        const int output_dim = KANQ_LAYER_OUTPUT_DIMS[layer];
        const float input_amax = KANQ_LAYER_INPUT_AMAX[layer];

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            next[output_index] = 0.0f;
        }

        for (int input_index = 0; input_index < input_dim; ++input_index) {
            const int slot = kanq_layer_slot(layer, input_index);
            const float x_q = kanq_fake_quant(current[input_index], input_amax);
            const int lut_index = kanq_lut_index(slot, x_q);
            const int lut_offset = slot * KANQ_LUT_SIZE + lut_index;
            const int basis_offset = lut_offset * KANQ_NUM_LOCAL_BASIS;
            const uint8_t first_cp = KANQ_LAYER_BASIS_FIRST_CP[lut_offset];
            const int16_t *restrict basis = &KANQ_LAYER_BASIS_PACK[basis_offset];
            const int16_t base_q = KANQ_LAYER_BASE_LUT[lut_offset];

            for (int output_index = 0; output_index < output_dim; ++output_index) {
                const int edge_slot = kanq_edge_slot(layer, input_index, output_index);
                const int coeff_offset =
                    ((edge_slot * KANQ_NUM_CONTROL_POINTS) + (int)first_cp) * KANQ_NUM_LOCAL_BASIS;
                const int16_t *restrict coeff4 = &KANQ_LAYER_COEFF4_PACK[coeff_offset];

                const int64_t spline_acc =
                    (int64_t)basis[0] * (int64_t)coeff4[0] +
                    (int64_t)basis[1] * (int64_t)coeff4[1] +
                    (int64_t)basis[2] * (int64_t)coeff4[2] +
                    (int64_t)basis[3] * (int64_t)coeff4[3];
                float contribution = (float)spline_acc * KANQ_LAYER_COEFF_DEQUANT[edge_slot];

                if (KANQ_BASE_BRANCH_ENABLED) {
                    contribution +=
                        (float)((int32_t)base_q * (int32_t)KANQ_LAYER_BASE_WEIGHT[edge_slot]) *
                        KANQ_LAYER_BASE_DEQUANT[edge_slot];
                }

                next[output_index] += contribution;
            }
        }

        for (int index = 0; index < output_dim; ++index) {
            current[index] = next[index];
            next[index] = 0.0f;
        }
    }

    return current[0];
}

float kan_quant_infer(float x) {
    float input[KANQ_INPUT_DIM] = {0.0f};
    input[0] = x;
    return kan_quant_infer_vector(input);
}


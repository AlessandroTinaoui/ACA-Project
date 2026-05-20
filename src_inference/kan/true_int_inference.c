#include "kan_true_int_inference.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "kan_model_true_int.h"

#ifndef KATI_HAS_IOC_CONTROL_LAYOUT
#define KATI_HAS_IOC_CONTROL_LAYOUT 0
#endif

static inline int kati_clamp_int(int value, int lower, int upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static inline int kati_clamp_i64_to_i32(int64_t value) {
    if (value < (int64_t)INT32_MIN) {
        return INT32_MIN;
    }
    if (value > (int64_t)INT32_MAX) {
        return INT32_MAX;
    }
    return (int)value;
}

static inline int kati_quantize_float_to_q(float value, float step) {
    if (step <= 0.0f) {
        return 0;
    }
    const int quantized = (int)lroundf(value / step);
    return kati_clamp_int(quantized, -KATI_QMAX, KATI_QMAX);
}

static inline int kati_basis_index(int layer, int input_q) {
    const int index = input_q + KATI_QMAX;
    if (index < 0) {
        return layer * KATI_BASIS_LUT_ENTRIES;
    }
    if (index >= KATI_BASIS_LUT_ENTRIES) {
        return layer * KATI_BASIS_LUT_ENTRIES + (KATI_BASIS_LUT_ENTRIES - 1);
    }
    return layer * KATI_BASIS_LUT_ENTRIES + index;
}

static inline int16_t kati_requantize_accumulator(int64_t acc) {
    const int shifted = kati_clamp_i64_to_i32((acc + KATI_REQUANT_ROUND) >> KATI_REQUANT_SHIFT);
    return (int16_t)kati_clamp_int(shifted, -KATI_QMAX, KATI_QMAX);
}

static inline int16_t kati_control_point_at(const int16_t *restrict cp_q, int index) {
    if (index < 0 || index >= KATI_NUM_CONTROL_POINTS) {
        return 0;
    }
    return cp_q[index];
}

static inline int64_t kati_dot4_i16(
    const int16_t *restrict basis_q15,
    int16_t coeff0,
    int16_t coeff1,
    int16_t coeff2,
    int16_t coeff3) {
    return
        (int64_t)basis_q15[0] * (int64_t)coeff0 +
        (int64_t)basis_q15[1] * (int64_t)coeff1 +
        (int64_t)basis_q15[2] * (int64_t)coeff2 +
        (int64_t)basis_q15[3] * (int64_t)coeff3;
}

#if KATI_HAS_IOC_CONTROL_LAYOUT
static inline int64_t kati_dot4_ioc_controls(
    const int16_t *restrict basis_q15,
    const int16_t *restrict control_q,
    int first_cp) {
    if (first_cp >= 0 && first_cp + KATI_NUM_LOCAL_BASIS <= KATI_NUM_CONTROL_POINTS) {
        const int16_t *restrict coeff4 = &control_q[first_cp];
        return kati_dot4_i16(basis_q15, coeff4[0], coeff4[1], coeff4[2], coeff4[3]);
    }

    return kati_dot4_i16(
        basis_q15,
        kati_control_point_at(control_q, first_cp + 0),
        kati_control_point_at(control_q, first_cp + 1),
        kati_control_point_at(control_q, first_cp + 2),
        kati_control_point_at(control_q, first_cp + 3));
}
#endif

float kan_true_int_infer_vector(const float *restrict input) {
    int16_t current_q[KATI_MAX_LAYER_WIDTH] = {0};
    int16_t next_q[KATI_MAX_LAYER_WIDTH] = {0};

    for (int index = 0; index < KATI_INPUT_DIM; ++index) {
        current_q[index] = (int16_t)kati_quantize_float_to_q(input[index], KATI_LAYER_INPUT_STEP[0]);
    }

    for (int layer = 0; layer < KATI_NUM_LAYERS; ++layer) {
        const int input_dim = KATI_LAYER_INPUT_DIMS[layer];
        const int output_dim = KATI_LAYER_OUTPUT_DIMS[layer];
        int64_t acc[KATI_MAX_OUTPUT_DIM] = {0};

        for (int input_index = 0; input_index < input_dim; ++input_index) {
            const int input_q = (int)current_q[input_index];
            const int lut_index = kati_basis_index(layer, input_q);
            const int basis_offset = lut_index * KATI_NUM_LOCAL_BASIS;
            const int first_cp = (int)KATI_LAYER_BASIS_FIRST_CP[lut_index];
            const int edge_base = KATI_LAYER_EDGE_OFFSETS[layer] + input_index * output_dim;
            const int16_t *restrict basis_q15 = &KATI_LAYER_BASIS_Q15[basis_offset];
#if KATI_BASE_BRANCH_ENABLED
            const int16_t base_q = KATI_LAYER_BASE_LUT_Q[lut_index];
#endif

#if KATI_HAS_IOC_CONTROL_LAYOUT
            const int control_base = edge_base * KATI_NUM_CONTROL_POINTS;

            for (int output_index = 0; output_index < output_dim; ++output_index) {
                const int edge_index = edge_base + output_index;
                const int16_t *restrict control_q =
                    &KATI_EDGE_CONTROL_POINTS_IOC_Q[control_base + output_index * KATI_NUM_CONTROL_POINTS];
                const int64_t spline_dot = kati_dot4_ioc_controls(basis_q15, control_q, first_cp);
                acc[output_index] += spline_dot * KATI_EDGE_CONTROL_MUL[edge_index];

#if KATI_BASE_BRANCH_ENABLED
                acc[output_index] +=
                    (int64_t)base_q *
                    (int64_t)KATI_EDGE_SCALE_BASE_Q[edge_index] *
                    KATI_EDGE_SCALE_BASE_MUL[edge_index];
#endif
            }
#else
            for (int output_index = 0; output_index < output_dim; ++output_index) {
                const int edge_index = edge_base + output_index;
                const int64_t spline_dot = kati_dot4_i16(
                    basis_q15,
                    kati_control_point_at(KATI_EDGE_CONTROL_POINTS_Q[edge_index], first_cp + 0),
                    kati_control_point_at(KATI_EDGE_CONTROL_POINTS_Q[edge_index], first_cp + 1),
                    kati_control_point_at(KATI_EDGE_CONTROL_POINTS_Q[edge_index], first_cp + 2),
                    kati_control_point_at(KATI_EDGE_CONTROL_POINTS_Q[edge_index], first_cp + 3));
                acc[output_index] += spline_dot * KATI_EDGE_CONTROL_MUL[edge_index];

#if KATI_BASE_BRANCH_ENABLED
                acc[output_index] +=
                    (int64_t)base_q *
                    (int64_t)KATI_EDGE_SCALE_BASE_Q[edge_index] *
                    KATI_EDGE_SCALE_BASE_MUL[edge_index];
#endif
            }
#endif
        }

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            next_q[output_index] = kati_requantize_accumulator(acc[output_index]);
        }

        if (layer + 1 < KATI_NUM_LAYERS) {
            for (int index = 0; index < output_dim; ++index) {
                current_q[index] = next_q[index];
                next_q[index] = 0;
            }
        }
    }

    return (float)next_q[0] * KATI_FINAL_OUTPUT_STEP;
}

float kan_true_int_infer(float x) {
    float input[KATI_INPUT_DIM] = {0.0f};
    input[0] = x;
    return kan_true_int_infer_vector(input);
}

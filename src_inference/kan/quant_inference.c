#include "kan_quant_inference.h"

#include <math.h>

#include "bspline.h"
#include "kan_model_quant.h"

static inline float kanq_base_silu(float x) {
    return x / (1.0f + expf(-x));
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

static inline float kanq_dot_uniform_cubic_basis_q(
    const int16_t *restrict control_points_q,
    float control_dequant,
    const BsplineCubicLocalBasis *restrict basis) {
    const int first_control_point = basis->first_control_point;
    float spline = 0.0f;

    for (int local_index = 0; local_index < 4; ++local_index) {
        const int cp_index = first_control_point + local_index;
        if (cp_index < 0 || cp_index >= KANQ_NUM_CONTROL_POINTS) {
            continue;
        }
        spline += (float)control_points_q[cp_index] * control_dequant * basis->value[local_index];
    }

    return spline;
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
            const float x_q = kanq_fake_quant(current[input_index], input_amax);
            const float base_value = kanq_base_silu(x_q);
            BsplineCubicLocalBasis basis = {0};
            const int has_spline = bspline_make_uniform_cubic_basis(
                x_q,
                KANQ_LAYER_KNOT_BASE[layer][input_index],
                KANQ_LAYER_KNOT_INV_DELTA[layer][input_index],
                KANQ_NUM_KNOTS,
                &basis);
            const int edge_base = KANQ_LAYER_EDGE_OFFSETS[layer] + input_index * output_dim;

            for (int output_index = 0; output_index < output_dim; ++output_index) {
                const int edge_index = edge_base + output_index;
                float contribution = 0.0f;

                if (has_spline) {
                    contribution = kanq_dot_uniform_cubic_basis_q(
                        KANQ_EDGE_CONTROL_POINTS[edge_index],
                        KANQ_EDGE_CONTROL_DEQUANT[edge_index],
                        &basis);
                }

                if (KANQ_BASE_BRANCH_ENABLED) {
                    contribution +=
                        (float)KANQ_EDGE_SCALE_BASE_Q[edge_index] *
                        KANQ_EDGE_SCALE_BASE_DEQUANT[edge_index] *
                        base_value;
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

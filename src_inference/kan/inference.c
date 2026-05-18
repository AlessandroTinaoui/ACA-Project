#include "kan_inference.h"

#include <math.h>

#include "bspline.h"
#include "kan_model.h"

#ifndef KAN_HAS_KNOT_METADATA
#define KAN_HAS_KNOT_METADATA 0
#endif

#ifndef KAN_USE_IMPLICIT_UNIFORM_KNOTS
#define KAN_USE_IMPLICIT_UNIFORM_KNOTS 0
#endif

#ifndef KAN_HAS_FLAT_EDGES
#define KAN_HAS_FLAT_EDGES 0
#endif

#ifndef KAN_BASE_BRANCH_ENABLED
#define KAN_BASE_BRANCH_ENABLED 0
#endif

#ifndef KAN_HAS_EDGE_MASKS
#define KAN_HAS_EDGE_MASKS 1
#endif

static inline float kan_base_silu(float x) {
    return x / (1.0f + expf(-x));
}

#if KAN_HAS_FLAT_EDGES
static inline int kan_edge_index(int layer, int input_index, int output_index, int output_dim) {
    return KAN_LAYER_EDGE_OFFSETS[layer] + input_index * output_dim + output_index;
}
#endif

static inline float kan_eval_spline(
    int layer,
    int input_index,
    float x,
    const float *restrict control_points) {
#if KAN_USE_IMPLICIT_UNIFORM_KNOTS
#if KAN_DEGREE == 3
    return bspline_eval_uniform_cubic_local(
        x,
        KAN_LAYER_KNOT_BASE[layer][input_index],
        KAN_LAYER_KNOT_INV_DELTA[layer][input_index],
        control_points,
        KAN_NUM_CONTROL_POINTS,
        KAN_NUM_KNOTS);
#else
    return bspline_eval_uniform(
        x,
        KAN_LAYER_KNOT_BASE[layer][input_index],
        KAN_LAYER_KNOT_DELTA[layer][input_index],
        control_points,
        KAN_DEGREE,
        KAN_NUM_CONTROL_POINTS,
        KAN_NUM_KNOTS);
#endif
#elif KAN_HAS_KNOT_METADATA
    if (KAN_LAYER_KNOTS_UNIFORM[layer][input_index]) {
#if KAN_DEGREE == 3
        return bspline_eval_uniform_cubic_local(
            x,
            KAN_LAYER_KNOT_BASE[layer][input_index],
            KAN_LAYER_KNOT_INV_DELTA[layer][input_index],
            control_points,
            KAN_NUM_CONTROL_POINTS,
            KAN_NUM_KNOTS);
#else
        return bspline_eval_uniform(
            x,
            KAN_LAYER_KNOT_BASE[layer][input_index],
            KAN_LAYER_KNOT_DELTA[layer][input_index],
            control_points,
            KAN_DEGREE,
            KAN_NUM_CONTROL_POINTS,
            KAN_NUM_KNOTS);
#endif
    }

    return bspline_eval(
        x,
        KAN_LAYER_KNOTS[layer][input_index],
        control_points,
        KAN_DEGREE,
        KAN_NUM_CONTROL_POINTS,
        KAN_NUM_KNOTS,
        KAN_X_MIN,
        KAN_X_MAX);
#else
    return bspline_eval(
        x,
        KAN_LAYER_KNOTS[layer][input_index],
        control_points,
        KAN_DEGREE,
        KAN_NUM_CONTROL_POINTS,
        KAN_NUM_KNOTS,
        KAN_X_MIN,
        KAN_X_MAX);
#endif
}

float kan_infer_vector(const float *restrict input) {
    float current[KAN_MAX_LAYER_WIDTH] = {0.0f};
    float next[KAN_MAX_LAYER_WIDTH] = {0.0f};

    for (int index = 0; index < KAN_INPUT_DIM; ++index) {
        current[index] = input[index];
    }

    for (int layer = 0; layer < KAN_NUM_LAYERS; ++layer) {
        const int input_dim = KAN_LAYER_INPUT_DIMS[layer];
        const int output_dim = KAN_LAYER_OUTPUT_DIMS[layer];

#if KAN_HAS_FLAT_EDGES && KAN_USE_IMPLICIT_UNIFORM_KNOTS && KAN_DEGREE == 3
        for (int output_index = 0; output_index < output_dim; ++output_index) {
            next[output_index] = 0.0f;
        }

        for (int input_index = 0; input_index < input_dim; ++input_index) {
            const float input_value = current[input_index];
            BsplineCubicLocalBasis basis = {0};
            const int has_spline = bspline_make_uniform_cubic_basis(
                input_value,
                KAN_LAYER_KNOT_BASE[layer][input_index],
                KAN_LAYER_KNOT_INV_DELTA[layer][input_index],
                KAN_NUM_KNOTS,
                &basis);
#if KAN_BASE_BRANCH_ENABLED
            const float base_value = kan_base_silu(input_value);
#endif
            const int edge_base = KAN_LAYER_EDGE_OFFSETS[layer] + input_index * output_dim;

            for (int output_index = 0; output_index < output_dim; ++output_index) {
                const int edge_index = edge_base + output_index;
#if KAN_HAS_EDGE_MASKS
                const float mask = KAN_EDGE_MASK[edge_index];
                if (mask == 0.0f) {
                    continue;
                }
#endif
                float edge_value = 0.0f;
                if (has_spline) {
                    edge_value +=
                        KAN_EDGE_SCALE_SP[edge_index] *
                        bspline_dot_uniform_cubic_basis(
                            KAN_EDGE_CONTROL_POINTS[edge_index],
                            KAN_NUM_CONTROL_POINTS,
                            &basis);
                }
#if KAN_BASE_BRANCH_ENABLED
                edge_value += KAN_EDGE_SCALE_BASE[edge_index] * base_value;
#endif
#if KAN_HAS_EDGE_MASKS
                next[output_index] += mask * edge_value;
#else
                next[output_index] += edge_value;
#endif
            }
        }

        for (int index = 0; index < output_dim; ++index) {
            current[index] = next[index];
            next[index] = 0.0f;
        }
        continue;
#endif

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            float sum = 0.0f;

            for (int input_index = 0; input_index < input_dim; ++input_index) {
#if KAN_HAS_FLAT_EDGES
                const int edge_index = kan_edge_index(layer, input_index, output_index, output_dim);
#if KAN_HAS_EDGE_MASKS
                const float mask = KAN_EDGE_MASK[edge_index];
                if (mask == 0.0f) {
                    continue;
                }
#endif

                const float input_value = current[input_index];
                const float spline = kan_eval_spline(
                    layer,
                    input_index,
                    input_value,
                    KAN_EDGE_CONTROL_POINTS[edge_index]);
                float edge_value = KAN_EDGE_SCALE_SP[edge_index] * spline;
#if KAN_BASE_BRANCH_ENABLED
                edge_value += KAN_EDGE_SCALE_BASE[edge_index] * kan_base_silu(input_value);
#endif
#if KAN_HAS_EDGE_MASKS
                sum += mask * edge_value;
#else
                sum += edge_value;
#endif
#else
                const float mask = KAN_LAYER_MASK[layer][input_index][output_index];
                if (mask == 0.0f) {
                    continue;
                }

                const float input_value = current[input_index];
                const float spline = kan_eval_spline(
                    layer,
                    input_index,
                    current[input_index],
                    KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index]);
                float edge_value =
                    KAN_LAYER_SCALE_SP[layer][input_index][output_index] * spline;
#if KAN_BASE_BRANCH_ENABLED
                edge_value +=
                    KAN_LAYER_SCALE_BASE[layer][input_index][output_index] *
                    kan_base_silu(input_value);
#endif

                sum += mask * edge_value;
#endif
            }

            next[output_index] = sum;
        }

        for (int index = 0; index < output_dim; ++index) {
            current[index] = next[index];
            next[index] = 0.0f;
        }
    }

    return current[0];
}

float kan_infer(float x) {
    float input[KAN_INPUT_DIM] = {0.0f};
    input[0] = x;
    return kan_infer_vector(input);
}

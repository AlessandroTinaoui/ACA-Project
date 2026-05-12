#include "kan_inference.h"

#include <math.h>

#include "bspline.h"
#include "kan_model.h"


float kan_infer(float x) {
    float current[KAN_MAX_LAYER_WIDTH] = {0.0f};
    float next[KAN_MAX_LAYER_WIDTH] = {0.0f};

    current[0] = x;

    for (int layer = 0; layer < KAN_NUM_LAYERS; ++layer) {
        const int input_dim = KAN_LAYER_INPUT_DIMS[layer];
        const int output_dim = KAN_LAYER_OUTPUT_DIMS[layer];

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            float sum = 0.0f;

            for (int input_index = 0; input_index < input_dim; ++input_index) {
                const float mask = KAN_LAYER_MASK[layer][input_index][output_index];
                if (fabsf(mask) == 0.0f) {
                    continue;
                }

                const float spline = bspline_eval(
                    current[input_index],
                    KAN_LAYER_KNOTS[layer][input_index],
                    KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index],
                    KAN_DEGREE,
                    KAN_NUM_CONTROL_POINTS,
                    KAN_NUM_KNOTS,
                    KAN_X_MIN,
                    KAN_X_MAX);

                float edge_value =
                    KAN_LAYER_SCALE_SP[layer][input_index][output_index] * spline;

                sum += mask * edge_value;
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

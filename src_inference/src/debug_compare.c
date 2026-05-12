#include <math.h>
#include <stdio.h>

#include "bspline.h"
#include "kan_inference.h"
#include "kan_model.h"
#include "mlp_inference.h"
#include "mlp_model.h"

#ifndef DEBUG_PI
#define DEBUG_PI 3.14159265358979323846f
#endif

static float target_function(float x) {
    return sinf(2.0f * DEBUG_PI * x) + 0.35f * sinf(10.0f * DEBUG_PI * x);
}

static void print_kan_trace(float x) {
    float current[KAN_MAX_LAYER_WIDTH] = {0.0f};
    float next[KAN_MAX_LAYER_WIDTH] = {0.0f};

    current[0] = x;
    printf("KAN_TRACE x=%.9g\n", x);

    for (int layer = 0; layer < KAN_NUM_LAYERS; ++layer) {
        const int input_dim = KAN_LAYER_INPUT_DIMS[layer];
        const int output_dim = KAN_LAYER_OUTPUT_DIMS[layer];

        printf("  layer=%d input_dim=%d output_dim=%d\n", layer, input_dim, output_dim);

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            float sum = 0.0f;

            for (int input_index = 0; input_index < input_dim; ++input_index) {
                const float mask = KAN_LAYER_MASK[layer][input_index][output_index];
                if (fabsf(mask) == 0.0f) {
                    continue;
                }

                const float input_value = current[input_index];
                const float *knots = KAN_LAYER_KNOTS[layer][input_index];
                const float *control_points =
                    KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index];
                const float spline = bspline_eval(
                    input_value,
                    knots,
                    control_points,
                    KAN_DEGREE,
                    KAN_NUM_CONTROL_POINTS,
                    KAN_NUM_KNOTS,
                    KAN_X_MIN,
                    KAN_X_MAX);
                const float edge_value =
                    KAN_LAYER_SCALE_SP[layer][input_index][output_index] * spline;
                const float contribution = mask * edge_value;

                printf(
                    "    edge in=%d out=%d x=%.9g spline=%.9g scale_sp=%.9g mask=%.9g contrib=%.9g\n",
                    input_index,
                    output_index,
                    input_value,
                    spline,
                    KAN_LAYER_SCALE_SP[layer][input_index][output_index],
                    mask,
                    contribution);
                printf("      bases:");
                for (int basis_index = 0; basis_index < KAN_NUM_CONTROL_POINTS; ++basis_index) {
                    const float basis = bspline_basis(
                        basis_index,
                        KAN_DEGREE,
                        input_value,
                        knots,
                        KAN_NUM_KNOTS,
                        KAN_NUM_CONTROL_POINTS,
                        KAN_X_MAX);
                    if (fabsf(basis) > 1.0e-6f) {
                        printf(
                            " i=%d knot=[%.9g,%.9g] B=%.9g c=%.9g cB=%.9g",
                            basis_index,
                            knots[basis_index],
                            knots[basis_index + KAN_DEGREE + 1],
                            basis,
                            control_points[basis_index],
                            control_points[basis_index] * basis);
                    }
                }
                printf("\n");

                sum += contribution;
            }

            next[output_index] = sum;
            printf("    output[%d]=%.9g\n", output_index, sum);
        }

        for (int index = 0; index < output_dim; ++index) {
            current[index] = next[index];
            next[index] = 0.0f;
        }
    }
}

int main(void) {
    static const float inputs[10] = {
        0.0f,
        1.0f / 9.0f,
        2.0f / 9.0f,
        3.0f / 9.0f,
        4.0f / 9.0f,
        5.0f / 9.0f,
        6.0f / 9.0f,
        7.0f / 9.0f,
        8.0f / 9.0f,
        1.0f,
    };

    printf("x,y_target,y_kan_c,y_mlp_c\n");
    for (int index = 0; index < 10; ++index) {
        const float x = inputs[index];
        const float input[MLP_INPUT_DIM] = {x};
        float output[MLP_OUTPUT_DIM] = {0.0f};

        mlp_infer(input, output);
        printf(
            "%.9g,%.9g,%.9g,%.9g\n",
            x,
            target_function(x),
            kan_infer(x),
            output[0]);
    }

    printf("\n");
    print_kan_trace(inputs[0]);
    print_kan_trace(inputs[5]);
    print_kan_trace(inputs[9]);

    return 0;
}

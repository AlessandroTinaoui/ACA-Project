#include "mlp_inference.h"

#include <math.h>

#include "mlp_model.h"

static float mlp_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float mlp_apply_activation(float x, int activation) {
    switch (activation) {
    case MLP_ACT_TANH:
        return tanhf(x);
    case MLP_ACT_RELU:
        return x > 0.0f ? x : 0.0f;
    case MLP_ACT_SIGMOID:
        return mlp_sigmoid(x);
    case MLP_ACT_IDENTITY:
    default:
        return x;
    }
}

void mlp_infer(const float *input, float *output) {
    float current[MLP_MAX_LAYER_WIDTH] = {0.0f};
    float next[MLP_MAX_LAYER_WIDTH] = {0.0f};
    int weight_offset = 0;
    int bias_offset = 0;

    for (int index = 0; index < MLP_INPUT_DIM; ++index) {
        current[index] = input[index];
    }

    for (int layer = 0; layer < MLP_NUM_LAYERS; ++layer) {
        const int input_dim = MLP_LAYER_SIZES[layer];
        const int output_dim = MLP_LAYER_SIZES[layer + 1];
        const int activation = MLP_LAYER_ACTIVATIONS[layer];

        for (int output_index = 0; output_index < output_dim; ++output_index) {
            float sum = MLP_BIASES[bias_offset + output_index];

            for (int input_index = 0; input_index < input_dim; ++input_index) {
                /*
                 * MLP_WEIGHTS is flattened row-major as:
                 * W[layer][out_neuron][in_neuron].
                 */
                const int weight_index =
                    weight_offset + output_index * input_dim + input_index;
                sum += MLP_WEIGHTS[weight_index] * current[input_index];
            }

            next[output_index] = mlp_apply_activation(sum, activation);
        }

        for (int index = 0; index < output_dim; ++index) {
            current[index] = next[index];
            next[index] = 0.0f;
        }

        weight_offset += input_dim * output_dim;
        bias_offset += output_dim;
    }

    for (int index = 0; index < MLP_OUTPUT_DIM; ++index) {
        output[index] = current[index];
    }
}

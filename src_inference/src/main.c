#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kan_inference.h"
#include "kan_model.h"

#ifndef KAN_PI
#define KAN_PI 3.14159265358979323846f
#endif

static float target_function(float x) {
    return sinf(2.0f * KAN_PI * x) + 0.35f * sinf(10.0f * KAN_PI * x);
}

static int parse_n(int argc, char **argv) {
    if (argc < 2) {
        return 1024;
    }

    char *end = NULL;
    const long value = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || value <= 0) {
        fprintf(stderr, "Invalid N: %s\n", argv[1]);
        return -1;
    }

    return (int)value;
}

int main(int argc, char **argv) {
    const int n = parse_n(argc, argv);
    if (n <= 0) {
        return 1;
    }

    double sum_squared_error = 0.0;
    double sum_abs_error = 0.0;
    double checksum_pred = 0.0;
    float max_abs_error = 0.0f;
    float last_x = 0.0f;
    float last_pred = 0.0f;
    float last_true = 0.0f;

    printf("KAN RISC-V demo\n");
    printf("N = %d\n", n);
    printf("num_layers = %d\n", KAN_NUM_LAYERS);
    printf("degree = %d\n", KAN_DEGREE);
    printf("num_control_points = %d\n", KAN_NUM_CONTROL_POINTS);
    printf("num_knots = %d\n", KAN_NUM_KNOTS);
    printf("x_min = %.9g\n", KAN_X_MIN);
    printf("x_max = %.9g\n", KAN_X_MAX);

    for (int i = 0; i < n; ++i) {
        const float x = (n == 1) ? KAN_X_MIN : (float)i / (float)(n - 1);
        const float y_pred = kan_infer(x);
        const float y_true = target_function(x);
        const float error = y_pred - y_true;
        const float abs_error = fabsf(error);

        sum_squared_error += (double)error * (double)error;
        sum_abs_error += (double)abs_error;
        checksum_pred += (double)y_pred;
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }

        if (i < 5) {
            printf("sample[%d]: x=%.9g y_pred=%.9g y_true=%.9g\n",
                   i,
                   x,
                   y_pred,
                   y_true);
        }

        last_x = x;
        last_pred = y_pred;
        last_true = y_true;
    }

    printf("last: x=%.9g y_pred=%.9g y_true=%.9g\n\n",
           last_x,
           last_pred,
           last_true);
    printf("MSE = %.9g\n", sum_squared_error / (double)n);
    printf("MAE = %.9g\n", sum_abs_error / (double)n);
    printf("MAX_ABS_ERROR = %.9g\n", max_abs_error);
    printf("CHECKSUM = %.9g\n", checksum_pred);
    printf("DONE\n");

    return 0;
}

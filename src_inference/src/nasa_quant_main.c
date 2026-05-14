#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gem5_stats.h"
#include "kan_model_quant.h"
#include "kan_quant_inference.h"
#include "nasa_test_data.h"

#ifndef KANQ_WARMUP_RUNS
#define KANQ_WARMUP_RUNS 1
#endif

static volatile float nasa_quant_benchmark_sink = 0.0f;

static int parse_n(int argc, char **argv) {
    if (argc < 2) {
        return NASA_TEST_SAMPLES;
    }

    char *end = NULL;
    const long value = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || value <= 0 || value > INT_MAX) {
        fprintf(stderr, "Invalid N: %s\n", argv[1]);
        return -1;
    }
    if (value > NASA_TEST_SAMPLES) {
        fprintf(stderr,
                "Invalid N: %ld, generated NASA header contains only %d samples\n",
                value,
                NASA_TEST_SAMPLES);
        return -1;
    }

    return (int)value;
}

static void run_warmup(int n) {
    float sink = 0.0f;

    for (int i = 0; i < KANQ_WARMUP_RUNS; ++i) {
        sink += kan_quant_infer_vector(NASA_TEST_FEATURES[i % n]);
    }

    nasa_quant_benchmark_sink = sink;
}

static void run_measured_inference(float *predictions, int n) {
    kan_gem5_reset_stats();

    for (int i = 0; i < n; ++i) {
        predictions[i] = kan_quant_infer_vector(NASA_TEST_FEATURES[i]) * KANQ_TARGET_SCALE;
    }

    kan_gem5_dump_stats();
}

int main(int argc, char **argv) {
    const int n = parse_n(argc, argv);
    if (n <= 0) {
        return 1;
    }
    if (NASA_TEST_INPUT_DIM != KANQ_INPUT_DIM) {
        fprintf(stderr,
                "NASA input dimension mismatch: data has %d, model expects %d\n",
                NASA_TEST_INPUT_DIM,
                KANQ_INPUT_DIM);
        return 1;
    }

    float *predictions = (float *)malloc((size_t)n * sizeof(*predictions));
    if (predictions == NULL) {
        fprintf(stderr, "Allocation failed for N = %d\n", n);
        return 1;
    }

    printf("NASA quantized KAN RISC-V demo\n");
    printf("N = %d\n", n);
    printf("available_samples = %d\n", NASA_TEST_SAMPLES);
    printf("input_dim = %d\n", KANQ_INPUT_DIM);
    printf("num_layers = %d\n", KANQ_NUM_LAYERS);
    printf("num_edges = %d\n", KANQ_NUM_EDGES);
    printf("degree = %d\n", KANQ_DEGREE);
    printf("num_control_points = %d\n", KANQ_NUM_CONTROL_POINTS);
    printf("num_knots = %d\n", KANQ_NUM_KNOTS);
    printf("num_intervals = %d\n", KANQ_NUM_INTERVALS);
    printf("quant_bits = %d\n", KANQ_BITS);
    printf("lut_size = %d\n", KANQ_LUT_SIZE);
    printf("target_scale = %.9g\n", KANQ_TARGET_SCALE);
    printf("measurement = kan_quant_infer_vector loop only\n");
    printf("warmup_runs = %d\n\n", KANQ_WARMUP_RUNS);
    fflush(stdout);

    run_warmup(n);
    run_measured_inference(predictions, n);

    double sum_squared_error = 0.0;
    double sum_abs_error = 0.0;
    double checksum_pred = 0.0;
    float max_abs_error = 0.0f;
#if NASA_HAS_REFERENCE_PREDICTIONS
    double sum_abs_reference_error = 0.0;
    float max_abs_reference_error = 0.0f;
#endif

    for (int i = 0; i < n; ++i) {
        const float y_pred = predictions[i];
        const float y_true = NASA_TEST_TARGETS[i];
        const float error = y_pred - y_true;
        const float abs_error = fabsf(error);

        sum_squared_error += (double)error * (double)error;
        sum_abs_error += (double)abs_error;
        checksum_pred += (double)y_pred;
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }

#if NASA_HAS_REFERENCE_PREDICTIONS
        const float reference_error = y_pred - NASA_TEST_REFERENCE_PREDICTIONS[i];
        const float abs_reference_error = fabsf(reference_error);
        sum_abs_reference_error += (double)abs_reference_error;
        if (abs_reference_error > max_abs_reference_error) {
            max_abs_reference_error = abs_reference_error;
        }
#endif

        if (i < 5) {
#if NASA_HAS_REFERENCE_PREDICTIONS
            printf("sample[%d]: y_pred=%.9g y_true=%.9g y_ref=%.9g\n",
                   i,
                   y_pred,
                   y_true,
                   NASA_TEST_REFERENCE_PREDICTIONS[i]);
#else
            printf("sample[%d]: y_pred=%.9g y_true=%.9g\n",
                   i,
                   y_pred,
                   y_true);
#endif
        }
    }

    printf("\nMSE = %.9g\n", sum_squared_error / (double)n);
    printf("RMSE = %.9g\n", sqrt(sum_squared_error / (double)n));
    printf("MAE = %.9g\n", sum_abs_error / (double)n);
    printf("MAX_ABS_ERROR = %.9g\n", max_abs_error);
    printf("CHECKSUM = %.9g\n", checksum_pred);
#if NASA_HAS_REFERENCE_PREDICTIONS
    printf("MAE_VS_REFERENCE = %.9g\n", sum_abs_reference_error / (double)n);
    printf("MAX_ABS_VS_REFERENCE = %.9g\n", max_abs_reference_error);
#endif
    printf("DONE\n");

    free(predictions);
    return 0;
}


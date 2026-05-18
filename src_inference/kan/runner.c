#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gem5_stats.h"
#include "rul_test_data.h"

#if defined(KAN_RUN_FP32) + defined(KAN_RUN_QUANT) + defined(KAN_RUN_TRUE_INT) != 1
#error "Define exactly one of KAN_RUN_FP32, KAN_RUN_QUANT, or KAN_RUN_TRUE_INT"
#endif

#if defined(KAN_RUN_FP32)
#include "kan_inference.h"
#include "kan_model.h"
#ifndef KAN_WARMUP_RUNS
#define KAN_WARMUP_RUNS 1
#endif
#define RUN_TITLE "KAN RISC-V demo"
#define RUN_INFER_VECTOR kan_infer_vector
#define RUN_INPUT_DIM KAN_INPUT_DIM
#define RUN_NUM_LAYERS KAN_NUM_LAYERS
#define RUN_NUM_EDGES KAN_NUM_EDGES
#define RUN_DEGREE KAN_DEGREE
#define RUN_NUM_CONTROL_POINTS KAN_NUM_CONTROL_POINTS
#define RUN_NUM_KNOTS KAN_NUM_KNOTS
#define RUN_NUM_INTERVALS KAN_NUM_INTERVALS
#define RUN_TARGET_SCALE KAN_TARGET_SCALE
#define RUN_WARMUP_RUNS KAN_WARMUP_RUNS
#define RUN_MEASUREMENT "kan_infer_vector loop only"
#define RUN_HAS_QUANT_BITS 0
#define RUN_PRINT_TARGET_SCALE 1
#elif defined(KAN_RUN_QUANT)
#include "kan_model_quant.h"
#include "kan_quant_inference.h"
#ifndef KANQ_WARMUP_RUNS
#define KANQ_WARMUP_RUNS 1
#endif
#define RUN_TITLE "Quantized KAN RISC-V demo"
#define RUN_INFER_VECTOR kan_quant_infer_vector
#define RUN_INPUT_DIM KANQ_INPUT_DIM
#define RUN_NUM_LAYERS KANQ_NUM_LAYERS
#define RUN_NUM_EDGES KANQ_NUM_EDGES
#define RUN_DEGREE KANQ_DEGREE
#define RUN_NUM_CONTROL_POINTS KANQ_NUM_CONTROL_POINTS
#define RUN_NUM_KNOTS KANQ_NUM_KNOTS
#define RUN_NUM_INTERVALS KANQ_NUM_INTERVALS
#define RUN_QUANT_BITS KANQ_BITS
#define RUN_TARGET_SCALE KANQ_TARGET_SCALE
#define RUN_WARMUP_RUNS KANQ_WARMUP_RUNS
#define RUN_MEASUREMENT "kan_quant_infer_vector loop only"
#define RUN_HAS_QUANT_BITS 1
#define RUN_PRINT_TARGET_SCALE 1
#else
#include "kan_model_true_int.h"
#include "kan_true_int_inference.h"
#ifndef KATI_WARMUP_RUNS
#define KATI_WARMUP_RUNS 1
#endif
#define RUN_TITLE "True-int KAN RISC-V demo"
#define RUN_INFER_VECTOR kan_true_int_infer_vector
#define RUN_INPUT_DIM KATI_INPUT_DIM
#define RUN_NUM_LAYERS KATI_NUM_LAYERS
#define RUN_NUM_EDGES KATI_NUM_EDGES
#define RUN_DEGREE KATI_DEGREE
#define RUN_NUM_CONTROL_POINTS KATI_NUM_CONTROL_POINTS
#define RUN_NUM_KNOTS KATI_NUM_KNOTS
#define RUN_NUM_INTERVALS KATI_NUM_INTERVALS
#define RUN_QUANT_BITS KATI_BITS
#define RUN_TARGET_SCALE KATI_TARGET_SCALE
#define RUN_WARMUP_RUNS KATI_WARMUP_RUNS
#define RUN_MEASUREMENT "kan_true_int_infer_vector loop only"
#define RUN_HAS_QUANT_BITS 1
#define RUN_PRINT_TARGET_SCALE 0
#endif

static volatile float benchmark_sink = 0.0f;

static int parse_n(int argc, char **argv) {
    if (argc < 2) {
        return TEST_SAMPLES;
    }

    char *end = NULL;
    const long value = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || value <= 0 || value > INT_MAX) {
        fprintf(stderr, "Invalid N: %s\n", argv[1]);
        return -1;
    }
    if (value > TEST_SAMPLES) {
        fprintf(stderr,
                "Invalid N: %ld, generated test header contains only %d samples\n",
                value,
                TEST_SAMPLES);
        return -1;
    }

    return (int)value;
}

static void run_warmup(int n) {
    float sink = 0.0f;

    for (int i = 0; i < RUN_WARMUP_RUNS; ++i) {
        sink += RUN_INFER_VECTOR(TEST_FEATURES[i % n]);
    }

    benchmark_sink = sink;
}

static void run_measured_inference(float *predictions, int n) {
    kan_gem5_reset_stats();

    for (int i = 0; i < n; ++i) {
        predictions[i] = RUN_INFER_VECTOR(TEST_FEATURES[i]) * RUN_TARGET_SCALE;
    }

    kan_gem5_dump_stats();
}

static void print_run_info(int n) {
    printf("%s\n", RUN_TITLE);
    printf("N = %d\n", n);
    printf("available_samples = %d\n", TEST_SAMPLES);
    printf("input_dim = %d\n", RUN_INPUT_DIM);
    printf("num_layers = %d\n", RUN_NUM_LAYERS);
    printf("num_edges = %d\n", RUN_NUM_EDGES);
    printf("degree = %d\n", RUN_DEGREE);
    printf("num_control_points = %d\n", RUN_NUM_CONTROL_POINTS);
    printf("num_knots = %d\n", RUN_NUM_KNOTS);
    printf("num_intervals = %d\n", RUN_NUM_INTERVALS);
#if RUN_HAS_QUANT_BITS
    printf("quant_bits = %d\n", RUN_QUANT_BITS);
#endif
#if RUN_PRINT_TARGET_SCALE
    printf("target_scale = %.9g\n", RUN_TARGET_SCALE);
#endif
    printf("measurement = %s\n", RUN_MEASUREMENT);
    printf("warmup_runs = %d\n\n", RUN_WARMUP_RUNS);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const int n = parse_n(argc, argv);
    if (n <= 0) {
        return 1;
    }
    if (TEST_INPUT_DIM != RUN_INPUT_DIM) {
        fprintf(stderr,
                "Input dimension mismatch: data has %d, model expects %d\n",
                TEST_INPUT_DIM,
                RUN_INPUT_DIM);
        return 1;
    }

    float *predictions = (float *)malloc((size_t)n * sizeof(*predictions));
    if (predictions == NULL) {
        fprintf(stderr, "Allocation failed for N = %d\n", n);
        return 1;
    }

    print_run_info(n);
    run_warmup(n);
    run_measured_inference(predictions, n);

    double sum_squared_error = 0.0;
    double sum_abs_error = 0.0;
    double checksum_pred = 0.0;
    float max_abs_error = 0.0f;
#if TEST_HAS_REFERENCE_PREDICTIONS
    double sum_abs_reference_error = 0.0;
    float max_abs_reference_error = 0.0f;
#endif

    for (int i = 0; i < n; ++i) {
        const float y_pred = predictions[i];
        const float y_true = TEST_TARGETS[i];
        const float error = y_pred - y_true;
        const float abs_error = fabsf(error);

        sum_squared_error += (double)error * (double)error;
        sum_abs_error += (double)abs_error;
        checksum_pred += (double)y_pred;
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }

#if TEST_HAS_REFERENCE_PREDICTIONS
        const float reference_error = y_pred - TEST_REFERENCE_PREDICTIONS[i];
        const float abs_reference_error = fabsf(reference_error);
        sum_abs_reference_error += (double)abs_reference_error;
        if (abs_reference_error > max_abs_reference_error) {
            max_abs_reference_error = abs_reference_error;
        }
#endif

        if (i < 5) {
#if TEST_HAS_REFERENCE_PREDICTIONS
            printf("sample[%d]: y_pred=%.9g y_true=%.9g y_ref=%.9g\n",
                   i,
                   y_pred,
                   y_true,
                   TEST_REFERENCE_PREDICTIONS[i]);
#else
            printf("sample[%d]: y_pred=%.9g y_true=%.9g\n", i, y_pred, y_true);
#endif
        }
    }

    printf("\nMSE = %.9g\n", sum_squared_error / (double)n);
    printf("RMSE = %.9g\n", sqrt(sum_squared_error / (double)n));
    printf("MAE = %.9g\n", sum_abs_error / (double)n);
    printf("MAX_ABS_ERROR = %.9g\n", max_abs_error);
    printf("CHECKSUM = %.9g\n", checksum_pred);
#if TEST_HAS_REFERENCE_PREDICTIONS
    printf("MAE_VS_REFERENCE = %.9g\n", sum_abs_reference_error / (double)n);
    printf("MAX_ABS_VS_REFERENCE = %.9g\n", max_abs_reference_error);
#endif
    printf("DONE\n");

    free(predictions);
    return 0;
}

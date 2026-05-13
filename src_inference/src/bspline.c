#include "bspline.h"

#include <math.h>

static inline float uniform_knot(float knot_base, float knot_delta, int index) {
    return knot_base + knot_delta * (float)index;
}

float bspline_basis(int i, int degree, float x, const float *knots, int num_knots, int num_control_points, float x_max) {
    if (degree == 0) {
        if (knots[i] <= x && x < knots[i + 1]) {
            return 1.0f;
        }

        if (x == x_max && i == num_control_points - 1 &&
            knots[i] <= x && x <= knots[i + 1] &&
            (i + 2 >= num_knots || knots[i + 2] == x_max)) {
            return 1.0f;
        }

        return 0.0f;
    }

    const float left_den = knots[i + degree] - knots[i];
    const float right_den = knots[i + degree + 1] - knots[i + 1];
    float left = 0.0f;
    float right = 0.0f;

    if (fabsf(left_den) > 0.0f) {
        left = ((x - knots[i]) / left_den) *
               bspline_basis(i, degree - 1, x, knots, num_knots, num_control_points, x_max);
    }
    if (fabsf(right_den) > 0.0f) {
        right = ((knots[i + degree + 1] - x) / right_den) *
                bspline_basis(i + 1, degree - 1, x, knots, num_knots, num_control_points, x_max);
    }

    return left + right;
}

float bspline_eval(float x, const float *knots, const float *control_points, int degree, int num_control_points, int num_knots, float x_min, float x_max) {
    (void)x_min;
    (void)x_max;

    float y = 0.0f;
    const int num_intervals = num_knots - 1;
    float prev[num_intervals];
    float curr[num_intervals];

    for (int i = 0; i < num_intervals; ++i) {
        prev[i] = (knots[i] <= x && x < knots[i + 1]) ? 1.0f : 0.0f;
    }

    for (int current_degree = 1; current_degree <= degree; ++current_degree) {
        const int basis_count = num_knots - current_degree - 1;

        for (int i = 0; i < basis_count; ++i) {
            const float left_den = knots[i + current_degree] - knots[i];
            const float right_den = knots[i + current_degree + 1] - knots[i + 1];
            float left = 0.0f;
            float right = 0.0f;

            if (fabsf(left_den) > 0.0f) {
                left = ((x - knots[i]) / left_den) * prev[i];
            }
            if (fabsf(right_den) > 0.0f) {
                right = ((knots[i + current_degree + 1] - x) / right_den) *
                        prev[i + 1];
            }

            curr[i] = left + right;
        }

        for (int i = 0; i < basis_count; ++i) {
            prev[i] = curr[i];
        }
    }

    for (int i = 0; i < num_control_points; ++i) {
        y += control_points[i] * prev[i];
    }

    return y;
}

float bspline_eval_uniform(float x, float knot_base, float knot_delta, const float *control_points, int degree, int num_control_points, int num_knots) {
    if (fabsf(knot_delta) <= 1.0e-12f) {
        return 0.0f;
    }

    float y = 0.0f;
    const int num_intervals = num_knots - 1;
    float prev[num_intervals];
    float curr[num_intervals];

    for (int i = 0; i < num_intervals; ++i) {
        const float left = uniform_knot(knot_base, knot_delta, i);
        const float right = uniform_knot(knot_base, knot_delta, i + 1);
        prev[i] = (left <= x && x < right) ? 1.0f : 0.0f;
    }

    for (int current_degree = 1; current_degree <= degree; ++current_degree) {
        const int basis_count = num_knots - current_degree - 1;

        for (int i = 0; i < basis_count; ++i) {
            const float knot_i = uniform_knot(knot_base, knot_delta, i);
            const float knot_i1 = uniform_knot(knot_base, knot_delta, i + 1);
            const float knot_i_d = uniform_knot(knot_base, knot_delta, i + current_degree);
            const float knot_i_d1 = uniform_knot(knot_base, knot_delta, i + current_degree + 1);
            const float left_den = knot_i_d - knot_i;
            const float right_den = knot_i_d1 - knot_i1;
            float left = 0.0f;
            float right = 0.0f;

            if (fabsf(left_den) > 0.0f) {
                left = ((x - knot_i) / left_den) * prev[i];
            }
            if (fabsf(right_den) > 0.0f) {
                right = ((knot_i_d1 - x) / right_den) * prev[i + 1];
            }

            curr[i] = left + right;
        }

        for (int i = 0; i < basis_count; ++i) {
            prev[i] = curr[i];
        }
    }

    for (int i = 0; i < num_control_points; ++i) {
        y += control_points[i] * prev[i];
    }

    return y;
}

#ifndef BSPLINE_H
#define BSPLINE_H

/*
 * Evaluate the B-spline basis B_{i,degree}(x).
 */
float bspline_basis(int i, int degree, float x, const float *knots, int num_knots, int num_control_points, float x_max);

/*
 * Evaluate sum_i control_points[i] * B_{i,degree}(x).
 */
float bspline_eval(float x, const float *knots, const float *control_points, int degree, int num_control_points, int num_knots, float x_min, float x_max);

/*
 * Evaluate the same spline for a uniform knot vector t_i = knot_base + i*knot_delta.
 */
float bspline_eval_uniform(float x, float knot_base, float knot_delta, const float *control_points, int degree, int num_control_points, int num_knots);

/*
 * Fast path for uniform cubic splines. At any x, at most degree+1 = 4 bases
 * are active, so this evaluates only the local basis quartet.
 */
typedef struct {
    int first_control_point;
    float value[4];
} BsplineCubicLocalBasis;

static inline int bspline_make_uniform_cubic_basis(
    float x,
    float knot_base,
    float knot_inv_delta,
    int num_knots,
    BsplineCubicLocalBasis *restrict basis) {
    if (knot_inv_delta <= 0.0f) {
        return 0;
    }

    const float position = (x - knot_base) * knot_inv_delta;
    if (position < 0.0f) {
        return 0;
    }

    const int span = (int)position;
    if (span >= num_knots - 1) {
        return 0;
    }

    const float u = position - (float)span;
    const float one_minus_u = 1.0f - u;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float one_minus_u2 = one_minus_u * one_minus_u;
    const float one_minus_u3 = one_minus_u2 * one_minus_u;
    basis->first_control_point = span - 3;
    basis->value[0] = one_minus_u3 * (1.0f / 6.0f);
    basis->value[1] = (3.0f * u3 - 6.0f * u2 + 4.0f) * (1.0f / 6.0f);
    basis->value[2] = (-3.0f * u3 + 3.0f * u2 + 3.0f * u + 1.0f) * (1.0f / 6.0f);
    basis->value[3] = u3 * (1.0f / 6.0f);
    return 1;
}

static inline float bspline_dot_uniform_cubic_basis(
    const float *restrict control_points,
    int num_control_points,
    const BsplineCubicLocalBasis *restrict basis) {
    const int first_control_point = basis->first_control_point;
    float y = 0.0f;

    if (first_control_point >= 0 && first_control_point < num_control_points) {
        y += control_points[first_control_point] * basis->value[0];
    }
    if (first_control_point + 1 >= 0 && first_control_point + 1 < num_control_points) {
        y += control_points[first_control_point + 1] * basis->value[1];
    }
    if (first_control_point + 2 >= 0 && first_control_point + 2 < num_control_points) {
        y += control_points[first_control_point + 2] * basis->value[2];
    }
    if (first_control_point + 3 >= 0 && first_control_point + 3 < num_control_points) {
        y += control_points[first_control_point + 3] * basis->value[3];
    }

    return y;
}

static inline float bspline_eval_uniform_cubic_local(
    float x,
    float knot_base,
    float knot_inv_delta,
    const float *restrict control_points,
    int num_control_points,
    int num_knots) {
    BsplineCubicLocalBasis basis;
    if (!bspline_make_uniform_cubic_basis(x, knot_base, knot_inv_delta, num_knots, &basis)) {
        return 0.0f;
    }
    return bspline_dot_uniform_cubic_basis(control_points, num_control_points, &basis);
}

#endif /* BSPLINE_H */

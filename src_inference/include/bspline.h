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

#endif /* BSPLINE_H */

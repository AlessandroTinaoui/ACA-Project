#ifndef KAN_INFERENCE_H
#define KAN_INFERENCE_H

/*
    Scalar wrapper kept for the original 1D synthetic demos.
 */
float kan_infer(float x);

/*
    General KAN inference for models with KAN_INPUT_DIM input features.
 */
float kan_infer_vector(const float *restrict input);

#endif /* KAN_INFERENCE_H */

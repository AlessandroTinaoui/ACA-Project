#ifndef KAN_QUANT_INFERENCE_H
#define KAN_QUANT_INFERENCE_H

/*
    Scalar wrapper kept for the original 1D synthetic demos.
 */
float kan_quant_infer(float x);

/*
    Quantized KAN inference for models with KANQ_INPUT_DIM input features.
 */
float kan_quant_infer_vector(const float *restrict input);

#endif /* KAN_QUANT_INFERENCE_H */


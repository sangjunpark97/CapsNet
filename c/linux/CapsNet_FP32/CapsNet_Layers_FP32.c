#include "capsnet.h"

inline void convolution(float* input, float* kernel, float* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride) {
    int i, j, k, p, l, m;
    for (i = 0; i < output_width; i++) {
        for (j = 0; j < output_width; j++) {
            for (k = 0; k < output_ch; k++) {
                output[k * output_width * output_width + i * output_width + j] = 0;
                for (p = 0; p < input_ch; p++) {
                    for (l = 0; l < kernel_width; l++) {
                        for (m = 0; m < kernel_width; m++) {
                            output[k * output_width * output_width + i * output_width + j] +=
                                input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)] *
                                kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m];
                        }
                    }
                }
            }
        }
    }
}

inline void squash(float* input, float* output, int dim_caps, int num_caps) {
    float tmp;
    for (int i=0; i < dim_caps; i++) {
        tmp = 0;
        for (int j=0; j < num_caps; j++) {
            tmp += pow(input[i + j * dim_caps], 2);
        }
        for (int k=0; k < num_caps; k++) {
            output[i + k * dim_caps] = (input[i + k * dim_caps] * sqrt(tmp)) / (tmp + 1);
        }
    }
}

inline void ReLU(float* input) {
    for (int i=0; i < conv1_out_ch * conv1_out_width * conv1_out_width; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}

inline void prediction_vectors(float* input, float* weight_matrix, float* output) {
    for (int l = 0; l < num_primary_caps; l++) {
        for (int k = 0; k < num_class; k++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] = 0;
                for (int i = 0; i < dim_primary_caps; i++) {
                    output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] +=
                    weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i] *
                    input[i + l * dim_primary_caps];
                }
            }
        }
    }
}
inline void softmax(float* input, float* output) {
    float tmp;
    for (int i = 0; i < num_class; i++) {
        tmp = 0;
        for (int j = 0;j < num_primary_caps; j++) {
            tmp += exp(input[j * num_class + i]);
        }
        for (int k = 0;k < num_primary_caps; k++) {
            output[k * num_class + i] = exp(input[k * num_class + i]) / tmp;
        }
    }
}

inline void dynamic_routing(float* uhat, float* v_j) {
    float c_ij[num_primary_caps * num_class];
    float b_ij[num_primary_caps * num_class];
    float s_j[num_class * dim_predic_vector];
    float tmp;
    int i, j, k, l;
    for (i = 0;i < num_primary_caps * num_class;i++) {
        b_ij[i] = 0;
    }
    for (i = 0;i < num_iterations;i++) {
        softmax(b_ij, c_ij);
        for (j = 0;j < num_class;j++) {
            for (k = 0;k < dim_predic_vector;k++) {
                tmp = 0;
                for (l = 0;l < num_primary_caps;l++) {
                    tmp += c_ij[num_class * l + j] * uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
                }
                s_j[j * dim_predic_vector + k] = tmp;
            }
        }
        squash(s_j, v_j, dim_predic_vector, num_class);
        for (j = 0; j < num_primary_caps; j++) {
            for (k = 0; k < num_class; k++) {
                tmp = 0;
                for (l = 0; l < dim_predic_vector; l++) {
                    tmp += uhat[j * num_class * dim_predic_vector + k * dim_predic_vector + l] * v_j[k * dim_predic_vector + l];
                }
                b_ij[num_class * j + k] += tmp;
            }
        }   
    }
}
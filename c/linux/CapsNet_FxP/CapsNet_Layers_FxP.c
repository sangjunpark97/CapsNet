#include "capsnet.h"
#include "fixedpoint.h"
inline void convolution(char* input, char* kernel, char* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride, int sq) {
    short tmp;
    int i, j, k, p, l, m;
    for (i = 0; i < output_width; i++) {
        for (j = 0; j < output_width; j++) {
            for (k = 0; k < output_ch; k++) {
                tmp = 0;
                for (p = 0; p < input_ch; p++) {
                    for (l = 0; l < kernel_width; l++) {
                        for (m = 0; m < kernel_width; m++) {
                            tmp = add_16(tmp, mul_16(input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)],
                                kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m]));
                            output[k * output_width * output_width + i * output_width + j] +=
                                input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)] *
                                kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m];
                        }
                    }
                }
                if (sq) {
                    output[k * output_width * output_width + i * output_width + j] = CLIP_8(tmp >> 7);
                }
                else {
                    if (tmp < 0) output[k * output_width * output_width + i * output_width + j] = 0;
                    else        output[k * output_width * output_width + i * output_width + j] = CLIP_8(tmp);
                }
            }
        }
    }
}

inline void prediction_vectors(char* input, char* weight_matrix, short* output) {
    for (int l = 0; l < num_primary_caps; l++) {
        for (int k = 0; k < num_class; k++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] = 0;
                for (int i = 0; i < dim_primary_caps; i++) {
                    output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] =
                        add_16(output[j + dim_predic_vector * k + dim_predic_vector * num_class * l],
                            mul_16(weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i],
                                input[i + l * dim_primary_caps]));
                }
            }
        }
    }
}

inline void dynamic_routing(short* uhat, char* v_j) {
    short c_ij[num_primary_caps * num_class];
    short b_ij[num_primary_caps * num_class];
    short s_j[num_class * dim_predic_vector];
    int tmp = 0;
    short int L1[16];
    short int L8[16];
    short int L2[16];
    char a_ = CLIP_8(fp2fx(0.448F));
    char b_ = CLIP_8(fp2fx(0.450F));
    int i, j, k, l;
    for (i = 0;i < num_primary_caps * num_class;i++) {
        b_ij[i] = 0;
    }
    for (i = 0;i < num_iterations;i++) {
        for (j = 0;j < num_class;j++) {
            for (k = 0;k < dim_predic_vector;k++) {
                for (l = 0;l < num_primary_caps;l++) {
                    tmp += (int)uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
                }
                s_j[j * dim_predic_vector + k] = CLIP_16(tmp >> fraction);
                tmp = 0;
            }
        }
        for (int j = 0; j < dim_predic_vector; j++) {
            L1[j] = 0; L2[j] = 0; L8[j] = 0;
            for (int k = 0; k < num_class; k++) {
                L1[j] = add_16(L1[j], mul_16(a_, qabs(s_j[k * dim_predic_vector + j])));
                if (qabs(s_j[k * dim_predic_vector + j]) > L8[j]) L8[j] = qabs(s_j[k * dim_predic_vector + j]);
            }
            tmp = mul_16(b_, L8[j]);
            L2[j] = add_16(L1[j], tmp);
            for (int k = 0; k < num_class; k++) {
                v_j[j + k * dim_predic_vector] = CLIP_8((s_j[j + k * dim_predic_vector] << fraction) / L2[j]);
            }
        }
        for (j = 0; j < num_primary_caps; j++) {
            for (k = 0; k < num_class; k++) {
                tmp = 0;
                for (l = 0; l < dim_predic_vector; l++) {
                    tmp = add_16(tmp, mul_16(v_j[k * dim_predic_vector + l], uhat[j * num_class * dim_predic_vector + k * dim_predic_vector + l]));
                }
                b_ij[num_class * j + k] += tmp;
            }
        }
    }
}
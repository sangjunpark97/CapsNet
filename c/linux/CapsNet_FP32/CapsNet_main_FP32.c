#include "capsnet.h"
#include "CapsNet_Layers_FP32.c"

float conv1_kernel[conv1_out_ch * conv1_in_ch * kernel_width * kernel_width];  
float conv2_kernel[conv2_out_ch * conv2_in_ch * kernel_width * kernel_width];
float conv1_output[conv1_out_ch * conv1_out_width * conv1_out_width];
float conv2_output[conv2_out_ch * conv2_out_width * conv2_out_width];
float digits_W[num_primary_caps * num_class * dim_predic_vector * dim_primary_caps];
float u_hat[num_primary_caps * num_class * dim_predic_vector];
float result_v[num_class * dim_predic_vector];

extern void convolution(float* input, float* kernel, float* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride);
extern void squash(float* input, float* output, int dim_caps, int num_caps);
extern void ReLU(float* input);
extern void prediction_vectors(float* input, float* weight_matrix, float* output);
extern void softmax(float* input, float* output);
extern void dynamic_routing(float* uhat, float* v_j);

int main() {
    FILE* fp, * num_wrong;
    int predict = 0;
    float max = 0;
    float accuracy;
    float predict_sum;
    int correct = 0;
    int tmp_predict;
    int wrong = 0;
    float dataset[input_data_ch * input_data_width * input_data_width];
    float LABEL[data_size];

    fp = fopen("label_float.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(LABEL, sizeof(float), data_size, fp);
    ///////////////////////////////////////////

    fp = fopen("conv1_kernel_16_float.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_kernel, sizeof(float), conv1_out_ch * conv1_in_ch * kernel_width * kernel_width, fp);
    ///////////////////////////////////////////

    fp = fopen("conv2_kernel_16_float.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_kernel, sizeof(float), conv2_out_ch * conv2_in_ch * kernel_width * kernel_width, fp);
    ///////////////////////////////////////////

    fp = fopen("digits_W_16_float.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_W, sizeof(float), num_primary_caps * num_class * dim_predic_vector * dim_primary_caps, fp);
    ///////////////////////////////////////////

    fp = fopen("mnist_float.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }

    for (int k = 0; k < data_size; k++) {
        fread(dataset, sizeof(float), input_data_ch * input_data_width * input_data_width, fp);
        convolution(dataset, conv1_kernel, conv1_output, conv1_in_width, conv1_in_ch, conv1_out_width, conv1_out_ch, conv1_stride);
        ReLU(conv1_output);
        ///////////////////////////////////////////

        convolution(conv1_output, conv2_kernel, conv2_output, conv2_in_width, conv2_in_ch, conv2_out_width, conv2_out_ch, conv2_stride);
        ///////////////////////////////////////////
        float conv2_output_transpose[num_primary_caps * dim_primary_caps];
        for (int i=0; i < dim_primary_caps; i++) {
            for (int j=0; j < num_primary_caps; j++) {
                conv2_output_transpose[j * dim_primary_caps + i] = conv2_output[i * num_primary_caps + j];
            }
        }
        squash(conv2_output_transpose,conv2_output,dim_primary_caps,num_primary_caps);
        prediction_vectors(conv2_output, digits_W, u_hat);
        ///////////////////////////////////////////
        dynamic_routing(u_hat, result_v);
        ///////////////////////////////////////////
        printf("\n");
        for (int i = 0; i < num_class; i++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                printf("%f, ", (result_v[j + i * dim_predic_vector]));
            }
            printf("\n");
        }
        max = 0;
        for (int i = 0; i < num_class; i++) {
            predict_sum = 0;
            for (int j = 0; j < dim_predic_vector; j++) {
                predict_sum += pow(result_v[i * dim_predic_vector + j], 2);
            }
            if (predict_sum > max) {
                max = predict_sum;
                predict = i;
            }
        }

        printf("(%d) pred : %d / ", k + 1, predict);
        if (predict == (int)*(LABEL + k))  correct++;
        printf("target: %d / ", (int)*(LABEL + k));
        accuracy = (float)correct / (k + 1);
        printf("accuracy : %.2f\n\n", accuracy * 100);

    }
    fclose(fp);
    return 0;
}
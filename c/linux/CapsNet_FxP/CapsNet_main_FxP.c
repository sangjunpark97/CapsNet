#include "capsnet.h"
#include <stdio.h>
#include "CapsNet_Layers_FxP.c"

char conv1_kernel[conv1_out_ch * conv1_in_ch * kernel_width * kernel_width];  
char conv2_kernel[conv2_out_ch * conv2_in_ch * kernel_width * kernel_width];
char conv1_output[conv1_out_ch * conv1_out_width * conv1_out_width];
char conv2_output[conv2_out_ch * conv2_out_width * conv2_out_width];
char digits_W[num_primary_caps * num_class * dim_predic_vector * dim_primary_caps];
short u_hat[num_primary_caps * num_class * dim_predic_vector];
char result_v[num_class * dim_predic_vector];

extern void convolution(char* input, char* kernel, char* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride, int sq);
extern void prediction_vectors(char* input, char* weight_matrix, short* output);
extern void dynamic_routing(short* uhat, char* v_j);

int main() {
    FILE* fp, * num_wrong;
    int predict = 0;
    int max = 0;
    float accuracy;
    int predict_sum;
    int correct = 0;
    int tmp_predict = 0;
    int wrong = 0;
    char dataset[input_data_ch * input_data_width * input_data_width];
    char LABEL[data_size];

    fp = fopen("label_char.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(LABEL, sizeof(char), data_size, fp);
    ///////////////////////////////////////////

    fp = fopen("conv1_kernel_16_char.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_kernel, sizeof(char), conv1_out_ch * conv1_in_ch * kernel_width * kernel_width, fp);
    ///////////////////////////////////////////

    fp = fopen("conv2_kernel_16_char.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_kernel, sizeof(char), conv2_out_ch * conv2_in_ch * kernel_width * kernel_width, fp);
    ///////////////////////////////////////////

    fp = fopen("digits_W_16_char.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_W, sizeof(char), num_primary_caps * num_class * dim_predic_vector * dim_primary_caps, fp);
    ///////////////////////////////////////////

    fp = fopen("mnist_char.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }

    for (int k = 0; k < data_size; k++) {
        fread(dataset, sizeof(char), input_data_ch * input_data_width * input_data_width, fp);
        printf("\n");
        convolution(dataset, conv1_kernel, conv1_output, conv1_in_width, conv1_in_ch, conv1_out_width, conv1_out_ch, conv1_stride, 0);
        convolution(conv1_output, conv2_kernel, conv2_output, conv2_in_width, conv2_in_ch, conv2_out_width, conv2_out_ch, conv2_stride, 1);
        ///////////////////////////////////////////
        char conv2_output_transpose[num_primary_caps * dim_primary_caps];
        for (int i = 0; i < dim_primary_caps; i++) {
            for (int j = 0; j < num_primary_caps; j++) {
                conv2_output_transpose[j * dim_primary_caps + i] = conv2_output[i * num_primary_caps + j];
            }
        }
        prediction_vectors(conv2_output_transpose, digits_W, u_hat);
        ///////////////////////////////////////////
        dynamic_routing(u_hat, result_v);
        ///////////////////////////////////////////
        printf("\n");
        for (int i = 0; i < num_class; i++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                printf("%d, ", (result_v[j + i * dim_predic_vector]));
            }
            printf("\n");
        }
        max = 0;
        for (int i = 0; i < num_class; i++) {
            predict_sum = 0;
            for (int j = 0; j < dim_predic_vector; j++) {
                tmp_predict = (result_v[i * dim_predic_vector + j] * result_v[i * dim_predic_vector + j]) >> (fraction);
                predict_sum = CLIP_16(predict_sum + tmp_predict);
            }
            if (predict_sum > max) {
                max = predict_sum;
                predict = i;
            }
        }
        printf("(%d) pred : %d / ", k + 1, predict);
        if (predict == *(LABEL + k))  correct++;
        printf("target: %d / ", *(LABEL + k));
        accuracy = (float)correct / (k + 1);
        printf("accuracy : %.2f\n\n", accuracy * 100);
    }
    fclose(fp);
    return 0;
}
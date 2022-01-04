#include "system.h"
#include "io.h"
#include <stdio.h>
#include "fixedpoint.h"
#include <sys/alt_cache.h>
#include "sys/alt_alarm.h"
#define input_BASE       	0x0
#define kernel0_BASE      	0x500000
#define kernel1_BASE     	0x1000000
#define sq_output_BASE 		0x700000
#define output_BASE     	0x3000000
#define csr_BASE         	0x04000000


void squash(char input[8 * 1152], char output[8 * 1152]) {
    int i, j, k;
    char tmp[8 * 1152];
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 32; j++) {
            for (k = 0; k < 36; k++) {
                tmp[i * 32 * 36 + j * 36 + k] = input[32 * 8 * k + i * 32 + j];
            }
        }
    }
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 1152; j++) {
            output[i * 1152 + j] = CLIP_16((tmp[j + 1152 * i] << fraction) / 127);
        }
    }

}

void u_hat_maker(char* input, char* W, short int* u_hat) {
    int i;
    int j;
    int k;
    short int tmp;


    for (i = 0; i < 1152 * 10 * 16; i++) {
        u_hat[i] = 0;
    }
    int l;
    for (l = 0; l < 1152; l++) {
        for (k = 0; k < 10; k++) {
            for (j = 0; j < 16; j++) {
                for (i = 0; i < 8; i++) {
                    tmp = ((W[l * 16 * 8 * 10 + 16 * 8 * k + 8 * j + i] * input[l * 8 * 10 + k * 8 + i]) >> fraction);
                    u_hat[j + 16 * k + 16 * 10 * l] = (u_hat[j + 16 * k + 16 * 10 * l] + tmp);
                }
            }
        }
    }// U_hat
}
void dr(short int* input, char* output) {
    
    int i, j, k;
    signed int tmp = 0;
    short int s_j[160];
    for (int i = 0; i < 160; i++) {
        s_j[i] = 0;
    }
    for (j = 0; j < 10; j++) {
        for (i = 0; i < 16; i++) {
            for (k = 0; k < 1152; k++) {
                tmp += input[i + j * 16 + 10 * 16 * k];
            }
            s_j[i + j * 16] = CLIP_16(tmp>>fraction);
            tmp = 0;
        }
    }
    int* s_ji = s_j;

    short int L1[16];
    short int L8[16];
    short int L2[16];
    fixedp a_ = CLIP_8(fp2fx(0.448F));
    fixedp b_ = CLIP_8(fp2fx(0.450F));
    for (int i = 0; i < 16; i++) {
        L1[i] = 0; L2[i] = 0; L8[i] = 0;
        for (int k = 0; k < 10; k++) {
            L1[i] = CLIP_16(L1[i] + CLIP_16((a_ * qabs(s_j[k * 16 + i]) >> fraction)));
            if (qabs(s_j[k * 16 + i]) > L8[i]) L8[i] = qabs(s_j[k * 16 + i]);
        }
        tmp = CLIP_16((b_ * L8[i]) >> fraction);
        L2[i] = CLIP_16(L1[i] + tmp);
        for (int j = 0; j < 10; j++) {
            output[i + j * 16] = CLIP_8((s_j[i + j * 16] << fraction) / L2[i]);
        }
    }

}

void csr(int o_c, int o_wh, int i_c, int i_wh,int S, int state){
	IOWR(csr_ptr,	20,	o_c		);
	IOWR(csr_ptr,	28,	o_wh	);
	IOWR(csr_ptr,	30,	i_c		);
	IOWR(csr_ptr,	38,	i_wh	);
	IOWR(csr_ptr,	40,	S		);
	IOWR(csr_ptr,	48, state	);
	IOWR(csr_ptr,	18, 	1	);
	IOWR(csr_ptr,	 8, 	1	); 
}


int main()
{
	volatile char* 	input_ptr 	 	= (char*)input_BASE;
    volatile char* 	kernel0_ptr 	= (char*)kernel0_BASE;
    volatile char* 	kernel1_ptr 	= (char*)kernel1_BASE;
    volatile char* 	output_ptr 	 	= (char*)output_BASE;
    volatile char* 	csr_ptr 		= (char*)csr_BASE;
    volatile char* 	sq_output_ptr 	= (char*)sq_output_BASE;
    volatile float* LABEL  = 0x0990000;
    volatile char*  buffer = 0x2500000;
	volatile char*  W_ptr  = 0x3700000;
	FILE* fp;
  FILE* FI;
  FILE* FL;
  int num 	= 0;
  int correct = 0;
	int max;
	int 	predict;
  float 	accuracy;

  fp = fopen("/mnt/host/fx_W_o_c16.bin", "rb"); 
	if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(W_ptr, sizeof(char), 1152 * 10 * 16 * 8, fp);
    fclose(fp);

	for (int i=0; i<6400; i++){
           IOWR(output_ptr, i , 0);
	}
	fclose(fp);

	fp = fopen("/mnt/host/fx_W_conv1_o_c16_re.bin", "rb"); if (fp == NULL)
    {
		printf("Cannot open file.\n");
		exit(1);
    }
	fread(buffer, sizeof(char), conv1_o_c * 81, fp);
	for(int i=0;  i<conv1_o_c * 81;i++){
		IOWR_8DIRECT(kernel0_ptr,i,buffer[i]);
	}
	fp = fopen("/mnt/host/fx_W_conv2_o_c16_re.bin", "rb"); 
	if (fp == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}
	
	for(int i=0;i<81;i++){
       fread(buffer, sizeof(char), conv1_o_c*256, fp);
       for(int j=0;  j<conv1_o_c * 256;j++){
           IOWR_8DIRECT(kernel1_ptr+i*conv1_o_c*256,j,buffer[j]);
       }
	}
	FL = fopen("/mnt/host/label_10000.bin", "rb"); 
	if (FL == NULL) {
       printf("Cannot open file.\n");
       exit(1);
	}

	FI = fopen("/mnt/host/fx_mnist_10000.bin", "rb"); 
	if (FI == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	for(int k=0; k<10000;k++){
		int start_time, finish_time, total_time;
		start_time = alt_nticks();
		fread(buffer, sizeof(char), 784, FI);
	
		for(int i=0;  i<784;i++){
			IOWR_8DIRECT(0x0,i,buffer[i]);
		}
	
		fread(LABEL, sizeof(float), 1, FL);
		csr(16, 20, 1, 28, 1, 0);

		while(1){
		if((IORD(csr_BASE,18)>>1)==1)break;
		}
  
		char tmp;
		for (int i = 0; i < 20 * 20 * conv1_o_c; i++) {
			tmp = IORD_8DIRECT(output_ptr, i);
			if(tmp>0)
				IOWR_8DIRECT(input_BASE, i, tmp);
			else
				IOWR_8DIRECT(input_BASE, i, 0);
		}//conv1 output conv2 input으로 복사
		
		csr(16, 20, 1, 28, 1, 1);

		while (1) {
			if ((IORD(csr_BASE, 18) >> 1) == 1) break;
		}

		squash(output_ptr, sq_output_ptr);
	
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 1152; j++) {
				input_ptr[j * 8 + i] = sq_output_ptr[i * 1152 + j];
			}
		}
		for (int i = 0; i < 1152; i++) {
			for (int j = 0; j < 10; j++) {
				for (int k = 0; k < 8; k++) {
					sq_output_ptr[i * 10 * 8 + j * 8 + k] = input_ptr[i * 8 + k];
				}
			}
		}
	
		u_hat_maker(sq_output_ptr, W_ptr, output_ptr);
		dr(output_ptr, input_ptr);
		
		int tmp_predict;
		int predict_sum[10];
		
		for (int i = 0; i < 10; i++) {
				predict_sum[i] = 0;
				for (int j = 0; j < 16; j++) {
					tmp_predict = ((input_ptr[j + i * 16] * input_ptr[j + i * 16]) >> fraction);
					predict_sum[i] = CLIP_16(predict_sum[i] + tmp_predict);
			}
		}
	
		float predict_sumf[10] = { 0 };
	
		for (int i = 0; i < 10; i++) {
			predict_sumf[i] = fx2fp(predict_sum[i]);
		}
    
		max = 0;
		for (int i = 0; i < 10; i++) {
			if (predict_sumf[i] >= max) {
				max = predict_sumf[i];
				predict = i;
			}
		}
		printf("(%d) pred : %d / ", k + 1, predict);
		if (predict == (int)*(LABEL)) correct++;
		printf("target: %d / ", (int)*(LABEL));
		accuracy = (float)correct / (k + 1);
		printf("accuracy : %.2f\n\n", accuracy * 100);
		finish_time = alt_nticks();
		total_time = ((finish_time - start_time) * 1000) /
		alt_ticks_per_second();
		printf("Conv time: %d ms\n", total_time);
    }
	return 0;
}

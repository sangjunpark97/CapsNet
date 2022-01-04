#include <stdio.h>
#include <math.h>
#include "vga_config.h"
#include "segment.h"
#include "altera_avalon_performance_counter.h"

#define datasize 10000

#define dataset_w 28
#define dataset_h 28

#define hor_num  = (int) ((SCREEN_WIDTH  - dataset_w * 2) / dataset_w) 
#define vert_num = (int) ((SCREEN_HEIGHT) / dataset_h)
#define page_num = (int) (hor_num * vert_num)

unsigned char mnist_8[784 * 10000];
float mnist_label[10000];

volatile int* hex54 = HEX4_5_BASE;
volatile int* hex30 = HEX0_3_BASE;
volatile int* ledr = LEDR_BASE;

int main(void) {

		FILE* fp;
	  	fp = fopen("/mnt/host/mnist_unsigned_char.bin", "rb");if (fp == NULL)
	  	{
	  		printf("Cannot open file.\n");
	  		exit(1);
	  	}

	  	fread(mnist_8, sizeof(unsigned char), 784 * 10000, fp);

        FILE* fp2;
	  	fp2 = fopen("/mnt/host/label_10000.bin", "rb");if (fp2 == NULL)
	  	{
	  		printf("Cannot open file.\n");
	  		exit(1);
	  	}

	  	fread(mnist_label, sizeof(float), 10000, fp2);


	vga_config();
	clear_screen();


	for(int u=0;u<140;u++){
		for(int m=0; m < vert_num; m++){
				for(int k=0; k < hor_num; k++){
					for(int i=0;i < dataset_h ;i++){
						for(int j=0; j < dataset_w; j++){
							plot_pixel(j + dataset_w * k, i + dataset_h * m, \
                            mnist_8[(dataset_w * dataset_h)*9*8*u + (dataset_w * dataset_h)*9*m + (dataset_w * dataset_h) * k + dataset_w * i + j]);
					}
				}
			}
		}
		clear_screen();
	}


    int page=0;
	for(int k = 0; k < datasize; k++){
            vga_disp(((k % page_num) % hor_num), ((k % page_num) / hor_num), (dataset_w * dataset_h) * k);
            page++;
            if(page == page_num) {
                clear_screen();
                page=0;
            }
			mnist_class_id_disp(mnist_label[k]);

            //////////inf code insert












            
            wrong_class();

	}
        
            
	}
}

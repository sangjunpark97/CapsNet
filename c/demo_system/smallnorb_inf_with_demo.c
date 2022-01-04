#include <stdio.h>
#include "system.h"
#include "demo_system.h"
#include <unistd.h>

#define dataset_w 32
#define dataset_h 32

#define datasize 100

#define hor_num   (int) ((SCREEN_WIDTH  - dataset_w * 2) / dataset_w)
#define vert_num  (int) ((SCREEN_HEIGHT) / dataset_h)
#define page_num  (int) (hor_num * vert_num)

volatile int* hex54 = HEX4_5_BASE;
volatile int* hex30 = HEX0_3_BASE;
volatile int* ledr = LEDR_BASE;

unsigned char smallnorb[32*32*48600];
float smallnorb_label[48600];

int main()
{
	printf("dataset load start\n");

	FILE* fp;
	fp = fopen("/mnt/host/smallnorb_unsigned_char.bin", "rb");if (fp == NULL)
	{
		 printf("Cannot open file.\n");
		 exit(1);
	}

	fread(smallnorb, sizeof(unsigned char), 32*32*100, fp);

	printf("dataset load fin\n");

	printf("label load start\n");

	FILE* fp2;
	fp2 = fopen("/mnt/host/label_smallnorb.bin", "rb");if (fp2 == NULL)
	{
		  printf("Cannot open file.\n");
		  exit(1);
	}

	fread(smallnorb_label, sizeof(float), 48600, fp2);

	printf("label load fin\n");

	vga_config();
	clear_screen();

	int page=0;
	for(int k=0; k < datasize; k++){
		vga_disp(((k % page_num) % hor_num), ((k % page_num) / hor_num), (dataset_w * dataset_h) * k);

		page++;
		if(page == page_num) {
			 clear_screen();
		 	page = 0;
		}
		smallNORB_class_id_disp(smallnorb_label[k]);
		  	    
	///////////////inf code insert








		  	  


	}

  return 0;
}

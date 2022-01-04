
#include "system.h"
#include "io.h"
#include <stdio.h>
#include <unistd.h>
#include "fixedpoint.h"
#include <sys/alt_cache.h>
#include "sys/alt_alarm.h"
#define word_s 16
#define conv1_o_c 16
#define CLIP_8(a) (a > 127? 127:(a < -128 ? -128: a))
#define fraction 4
#define input_BASE       0x0
#define kernel0_BASE      0x500000
#define kernel1_BASE      0x1000000
#define output_BASE      0x3000000
#define csr_BASE         0x04000000
#define mul_8(a,b) CLIP_8((a*b)>>fraction)
#define mul_16(a,b) CLIP_16((a*b)>>fraction)
#define add_8(a,b) CLIP_8(a + b)
#define add_16(a,b) CLIP_16(a + b)
volatile int* hex54 = HEX5_HEX4_BASE;
volatile int* hex30 = HEX3_HEX0_BASE;
volatile int* ledr = LEDR_BASE;
volatile unsigned char smallnorb[28*28];
volatile float smallnorb_label[1];

#define datasize 10000

#define dataset_w 28
#define dataset_h 28

#define hor_num   (int) ((SCREEN_WIDTH  - dataset_w * 2) / dataset_w)
#define vert_num  (int) ((SCREEN_HEIGHT) / dataset_h)
#define page_num  (int) (hor_num * vert_num)

#define PIXEL(r, g, b) \
   (short int)((((r)&0x1f)<<11)|(((g)&0x3f)<<5)|(((b)&0x1f)))

#define FILL_PIXEL(x,y,r,g,b)\
   *(short int *)(pixel_buffer_start + (((y)&0xff)<<10) + (((x)&0x1ff)<<1))=PIXEL(r,g,b)

#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240

volatile int pixel_buffer_start;
volatile int* pixel_ctrl_ptr;

volatile short int* front_buffer;
volatile short int* back_buffer;

void squash(char input[8 * 1152], char output[8 * 1152]) {
    for(int i=0;i<8*1152;i++){
        output[i]=input[i];
    }
}

void u_hat_maker(char input[8 * 1152], char W[1152 * 10 * 16 * 8], short int* u_hat) {
    for (int i = 0; i < 1152 * 10 * 16; i++) {
        u_hat[i] = 0;
    }

    for (int l = 0; l < 1152; l++) { // matmul
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < 16; j++) {
                for (int i = 0; i < 8; i++) {
                    u_hat[j + 16 * k + 16 * 10 * l]  = add_16(u_hat[j + 16 * k + 16 * 10 * l], mul_16(W[l * 16 * 8 * 10 + 16 * 8 * k + 8 * j + i], input[1152*i + l]));

                }
            }
        }
    }// U_hat     >>> u_hat[1152 * 10 * 16 * 1]
}

void iteration_3(short int* input, char* output) {
    /*2     iteration*/
    int i, j, k;
    signed int tmp = 0;
    short int s_j[160];
    // routing iteration 3?
    //char sum = CLIP_8(fp2fx(0.1)); //변환하면 1임
    //char sum1 = CLIP_8(fp2fx(0.086)); //변환하면 1임
    //printf("0.1 : %d 0.086 : %d \n", sum,sum1);
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

            //printf("u_hat2 %d, ", u_hat[i + j * 16 + 10 * 16 * k]);
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
            //printf("s_j : %f\n", fx2fp(s_j[k * 16 + i]));
            L1[i] = CLIP_16(L1[i] + CLIP_16((a_ * qabs(s_j[k * 16 + i]) >> fraction)));
            if (qabs(s_j[k * 16 + i]) > L8[i]) L8[i] = qabs(s_j[k * 16 + i]);
        }
        tmp = CLIP_16((b_ * L8[i]) >> fraction);
        L2[i] = CLIP_16(L1[i] + tmp);
        for (int j = 0; j < 10; j++) {
            output[i + j * 16] = CLIP_8((s_j[i + j * 16] << fraction) / L2[i]);
        }
        //printf("%f, %f\n", fx2fp(L1[i]), fx2fp(L2[i]));
    }

}
void wait_for_vsync() {
   register int status;
   *pixel_ctrl_ptr = 1;

   status = *(pixel_ctrl_ptr + 3);
   while ((status & 0x01) != 0)
      status = *(pixel_ctrl_ptr + 3);
}

void draw_square(int x1, int y1, int x2, int y2, int r, int g, int b) {
   int x, y;
   for (x = x1; x <= x2; x++) {
      for (y = y1; y <= y2; y++) {
         FILL_PIXEL(x, y, r, g, b);
      }
   }
}

/*void clear_screen(int r, int g, int b) {
   draw_square(0, 0, SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1, r, g, b);
}*/



void plot_pixel(int x, int y, unsigned char line_color){
   *(unsigned char *)(pixel_buffer_start + (y << 9) + (x)) = line_color;
}

void clear_screen(void) {
   for(int i=0;i<SCREEN_WIDTH;i++){
      for(int j=0;j<SCREEN_HEIGHT;j++){
         plot_pixel(i, j, 0x00);
      }
   }
}



void vga_config(void){
    pixel_ctrl_ptr = (int*)VGA_SUBSYSTEM_VGA_PIXEL_DMA_BASE;
   pixel_buffer_start = *pixel_ctrl_ptr;
   *(pixel_ctrl_ptr + 1) = front_buffer;
      wait_for_vsync();

      pixel_buffer_start = *pixel_ctrl_ptr;
      clear_screen();
      *(pixel_ctrl_ptr + 1) = back_buffer;

}

void vga_disp(int x_coordinate, int y_coordinate){
   for(int i = 0; i < dataset_h ; i++){
      for(int j = 0; j < dataset_w; j++){
         plot_pixel(j + x_coordinate * dataset_w, i + y_coordinate*dataset_h, smallnorb[ dataset_w * i + j]);
       }
   }
}
void wrong_class(void){
   *ledr = 0b1111111111;
}


volatile int segment_char(char alphabet) {
   volatile int seg_value;
   switch (alphabet) {
      case 'a':   seg_value = 119;
               break;
      case 'b':   seg_value = 124;
               break;
      case 'c':   seg_value = 57;
               break;
      case 'd':   seg_value = 94;
               break;
      case 'e':   seg_value = 121;
               break;
      case 'f':   seg_value = 113;
               break;
      case 'g':   seg_value = 61;
               break;
      case 'h':   seg_value = 116;
               break;
      case 'i':   seg_value = 4;
               break;
      case 'j':   seg_value = 14;
               break;
      case 'k':   seg_value = 117;
               break;
      case 'l':   seg_value = 56;
               break;
      case 'm':   seg_value = 85;
               break;
      case 'n':   seg_value = 84;
               break;
      case 'o':   seg_value = 92;
               break;
      case 'p':   seg_value = 115;
               break;
      case 'q':   seg_value = 103;
               break;
      case 'r':   seg_value = 80;
               break;
      case 's':   seg_value = 109;
               break;
      case 't':   seg_value = 120;
               break;
      case 'u':   seg_value = 62;
               break;
      case 'v':   seg_value = 28;
               break;
      case 'w':   seg_value = 106;
               break;
      case 'x':   seg_value = 118;
               break;
      case 'y':   seg_value = 110;
               break;
      case 'z':   seg_value = 91;
               break;
   }
   return seg_value;
}

volatile int segment_num(int num) {
   volatile int seg_value;
   switch (num) {
   case 0:      seg_value = 63;
            break;
   case 1:      seg_value = 6;
            break;
   case 2:      seg_value = 91;
            break;
   case 3:      seg_value = 79;
            break;
   case 4:      seg_value = 102;
            break;
   case 5:      seg_value = 105;
            break;
   case 6:      seg_value = 125;
            break;
   case 7:      seg_value = 7;
            break;
   case 8:      seg_value = 127;
            break;
   case 9:      seg_value = 111;
            break;
   }
   return seg_value;
}

void segment_str(char* str) {
   *hex54 = segment_char(*str) << 8 | segment_char(*(str + 1));
   *hex30 = segment_char(*(str + 2)) << 24 | segment_char(*(str + 3)) << 16 | segment_char(*(str + 4)) << 8 | segment_char(*(str + 5));
}

void mnist_class_id_disp(int mnist_label, int predict) {
	int seg_value = segment_num(mnist_label) | segment_num(predict)<<16;

	*hex30 = seg_value;
}

void smallNORB_class_id_disp(float smallNORB_label) {
    int tmp = (int)smallNORB_label;
    switch(tmp){
        case 0:     segment_str("animal");
                    break;
        case 1:     segment_str("humans");
                    break;
        case 2:     segment_str("planes");
                    break;
        case 3:     segment_str("trucks");
                    break;
        case 4:     segment_str("cars");
                    break;
    }
}
int main()
{
    volatile char* input_ptr = (char*)input_BASE;
    volatile char* kernel0_ptr = (char*)kernel0_BASE;
    volatile char* kernel1_ptr = (char*)kernel1_BASE;
    volatile char* output_ptr = (char*)output_BASE;
    volatile char* csr_ptr = (char*)csr_BASE;
    volatile float* LABEL =0x0990000;
    volatile char* buffer=0x2500000;
    volatile char* W_ptr=0x3700000;
    volatile char* dr_output_ptr=0x3400000;
    FILE* fp;
    FILE* FI;
    FILE* FL;
    int num    = 0;
    int correct = 0;
    int max;
    int    predict;
    float    accuracy;

    /*fp = fopen("/mnt/host/fx_W_o_c16.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(W_ptr, sizeof(char), 1152 * 10 * 16 * 8, fp);


    fclose(fp);

   //for (int i=0; i<6400; i++){
   //        IOWR(0x3000000, i , 0);
   //}
   //fclose(fp);

   fp = fopen("/mnt/host/fx_W_conv1_o_c16_rere.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
   fread(buffer, sizeof(char), conv1_o_c * 81, fp);
   for(int i=0;  i<conv1_o_c * 81;i++){
      IOWR_8DIRECT(0x500000,i,buffer[i]);
   }
   fp = fopen("/mnt/host/fx_W_conv2_o_c16_rere.bin", "rb"); if (fp == NULL)
   {
       printf("Cannot open file.\n");
       exit(1);
   }
   for(int i=0;i<conv1_o_c*256;i++){
       fread(buffer, sizeof(char), 81, fp);
       for(int j=0;  j< 81;j++){
           IOWR_8DIRECT(kernel1_ptr+i*81,j,buffer[j]);
       }
   }*/
   int start_time, finish_time, total_time;
       start_time = alt_nticks();
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
   fp = fopen("/mnt/host/mnist_unsigned_char.bin", "rb");if (fp == NULL)
           {
              printf("Cannot open file.\n");
              exit(1);
           }
   vga_config();
   clear_screen();
    for(int k=0; k<10000;k++){
      int start_time, finish_time, total_time;
        fread(buffer, sizeof(char), 784, FI);

      for(int i=0;  i<784;i++){
         IOWR_8DIRECT(0x0,i,buffer[i]);
      }
    fread(LABEL, sizeof(float), 1, FL);

           fread(smallnorb, sizeof(unsigned char), 28*28, fp);
           printf("dataset load fin\n");

           printf("label load start\n");

           FILE* fp2;
                   fp2 = fopen("/mnt/host/label_10000.bin", "rb");if (fp2 == NULL)
                   {
                      printf("Cannot open file.\n");
                      exit(1);
                   }

                   fread(smallnorb_label, sizeof(float), 1, fp2);
                printf("label load fin\n");



           int page=0;
              vga_disp(((k % page_num) % hor_num), ((k % page_num) / hor_num));



              page++;
               if(page == page_num) {
                  clear_screen();
                  page = 0;
               }
               //////inf code insert






//   fclose(fp);
   int a = 20;
   int b = 1;
   int c = 28;
   int d = 1;
   int e = 16;
   //convolution(input_ptr, kernel_ptr, output_ptr, 16, 20, 1, 28, 1);

   IOWR(0x04000020,0, e);
   IOWR(0x04000028,0, a);
   IOWR(0x04000030,0, b);
   IOWR(0x04000038,0, c);
   IOWR(0x04000040,0, d);

   IOWR(0x04000048, 0, 0);
   IOWR(0x4000018,0,0);
   IOWR(0x04000008,0,1); // start

   start_time = alt_nticks();

    while(1){
      if((IORD(0x4000018,0)>>1)==1)break;
   }

    finish_time = alt_nticks();
          total_time = ((finish_time - start_time) * 1000) /
          alt_ticks_per_second();
          printf("1D Conv + ReLu time: %d ms\n", total_time);
   char tmp;

   for (int i = 0; i < 20 * 20 * conv1_o_c; i++) {
       tmp = IORD_8DIRECT(output_ptr, i);
       IOWR_8DIRECT(input_BASE, i, tmp);
   }//conv1 output conv2 input으로 복사



   IOWR(0x04000020, 0, 256);
   IOWR(0x04000028, 0, 6);
   IOWR(0x04000030, 0, conv1_o_c);
   IOWR(0x04000038, 0, 20);
   IOWR(0x04000040, 0, 2);
   IOWR(0x04000048, 0, 1);

   IOWR(0x4000018,0,0);
   IOWR(0x04000008, 0, 1); // start


  // convolution(input_ptr, kernel_ptr, output_ptr, 256, 6, 16, 20, 2);


   start_time = alt_nticks();

   while (1) {
       if ((IORD(0x4000018, 0) >> 1) == 1)break;
   }

   finish_time = alt_nticks();
            total_time = ((finish_time - start_time) * 1000) /
            alt_ticks_per_second();
            printf("2D Conv time: %d ms\n", total_time);




   start_time = alt_nticks();
   squash(output_ptr, input_ptr);
   finish_time = alt_nticks();
               total_time = ((finish_time - start_time) * 1000) /
               alt_ticks_per_second();
               printf("Squash time: %d ms\n", total_time);


   //u_hat_maker(input_ptr, W_ptr, output_ptr);

                        IOWR(0x04000088, 0, 1); // start
                        start_time = alt_nticks();
                        while (1) {
                            if ((IORD(0x4000098, 0) >> 1) == 1)break;
                        }
   finish_time = alt_nticks();
                  total_time = ((finish_time - start_time) * 1000) /
                  alt_ticks_per_second();
                  printf("transpose + u_hat time: %d ms\n", total_time);
   start_time = alt_nticks();

   iteration_3(dr_output_ptr, input_ptr);
   finish_time = alt_nticks();
                     total_time = ((finish_time - start_time) * 1000) /
                     alt_ticks_per_second();
                     printf("dr time: %d ms\n", total_time);
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


    mnist_class_id_disp((int)*(LABEL) ,predict);


    }

   return 0;

}

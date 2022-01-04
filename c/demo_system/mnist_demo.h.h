#include "system.h"
#include <stdio.h>

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
extern volatile int* ledr;
extern volatile int* hex30;
extern volatile int* hex54;
extern unsigned char mnist_8[784 * 10000];

short int front_buffer[512 * 256];
short int back_buffer[512 * 256];

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

void vga_disp(int x_coordinate, int y_coordinate, int data_num){
	for(int i = 0; i < dataset_h ; i++){
		for(int j = 0; j < dataset_w; j++){    
			plot_pixel(j + x_coordinate * dataset_w, i + y_coordinate*dataset_h, mnist_8[data_num + dataset_w * i + j]);
	    }
	}
}

void wrong_class(void){
	*ledr = 0b1111111111;
}


volatile int segment_char(char alphabet) {
	volatile int seg_value;
	switch (alphabet) {
		case 'a':	seg_value = 119;
					break;
		case 'b':	seg_value = 124;
					break;
		case 'c':	seg_value = 57;
					break;
		case 'd':	seg_value = 94;
					break;
		case 'e':	seg_value = 121;
					break;
		case 'f':	seg_value = 113;
					break;
		case 'g':	seg_value = 61;
					break;
		case 'h':	seg_value = 116;
					break;
		case 'i':	seg_value = 4;
					break;
		case 'j':	seg_value = 14;
					break;
		case 'k':	seg_value = 117;
					break;
		case 'l':	seg_value = 56;
					break;
		case 'm':	seg_value = 85;
					break;
		case 'n':	seg_value = 84;
					break;
		case 'o':	seg_value = 92;
					break;
		case 'p':	seg_value = 115;
					break;
		case 'q':	seg_value = 103;
					break;
		case 'r':	seg_value = 80;
					break;
		case 's':	seg_value = 109;
					break;
		case 't':	seg_value = 120;
					break;
		case 'u':	seg_value = 62;
					break;
		case 'v':	seg_value = 28;
					break;
		case 'w':	seg_value = 106;
					break;
		case 'x':	seg_value = 118;
					break;
		case 'y':	seg_value = 110;
					break;
		case 'z':	seg_value = 91;
					break;			
	}
	return seg_value;
}

volatile int segment_num(int num) {
	volatile int seg_value;
	switch (num) {
	case 0:		seg_value = 63;
				break;
	case 1:		seg_value = 6;
				break;
	case 2:		seg_value = 91;
				break;
	case 3:		seg_value = 79;
				break;
	case 4:		seg_value = 102;
				break;
	case 5:		seg_value = 105;
				break;
	case 6:		seg_value = 125;
				break;
	case 7:		seg_value = 7;
				break;
	case 8:		seg_value = 127;
				break;
	case 9:		seg_value = 111;
				break;
	}
	return seg_value;
}

void segment_str(char* str) {
	*hex54 = segment_char(*str) << 8 | segment_char(*(str + 1));
	*hex30 = segment_char(*(str + 2)) << 24 | segment_char(*(str + 3)) << 16 | segment_char(*(str + 4)) << 8 | segment_char(*(str + 5));
}

void mnist_class_id_disp(float mnist_label) {
	int tmp = (int)mnist_label;
	int seg_value = segment_num(tmp);

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


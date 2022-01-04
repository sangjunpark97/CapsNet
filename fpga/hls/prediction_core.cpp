
#include "HLS/hls.h"
#include "assert.h"
#include "HLS/stdio.h"
#include <stdlib.h>
#define fraction 4
#define word_s 16
#define conv1_o_c 16
#define CLIP_8(a) (a > 127? 127:(a < -128 ? -128: a))
#define CLIP_16(a) (a >  32767?  32767:(a < -32768 ? -32768: a))
#define mul_8(a,b) CLIP_8((a*b)>>fraction)
#define mul_16(a,b) CLIP_16(((short int)a*(short int)b)>>fraction)
#define add_8(a,b) CLIP_8(a + b)
#define add_16(a,b) CLIP_16(a + b)

typedef ihc::mm_master<char, ihc::dwidth<16>,
 ihc::awidth<32>,
 ihc::aspace<1>,
 ihc::latency<0>,
 ihc::waitrequest<1>  > Master;


hls_avalon_slave_component component void dr
(Master &input0,
 Master &kernel0,
 Master &output0
 ) {
    int i;
    int j;
    int k;
    short int tmp;
    for (int i = 0; i < 1152 * 10 * 16; i++) {
        output0[i] = 0;
    }

    for (int l = 0; l < 1152; l++) { // matmul
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < 16; j++) {
                for (int i = 0; i < 8; i++) {
                   output0[j + 16 * k + 16 * 10 * l]  = add_16(output0[j + 16 * k + 16 * 10 * l], mul_16(kernel0[l * 16 * 8 * 10 + 16 * 8 * k + 8 * j + i], input0[1152*i + l]));
                }
            }
        }
    }
}
int main(){
	printf("done");
}	

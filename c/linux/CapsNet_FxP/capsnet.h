#ifndef _capsnet_h_
#define _capsnet_h_

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define data_size 10000
#define input_data_width 28
#define input_data_ch 1
//input data(input_data_width, input_data_width)
#define conv1_in_width input_data_width
#define conv1_in_ch input_data_ch
#define conv1_out_width 20
#define conv1_out_ch 16
#define conv1_stride 1
#define conv2_in_width conv1_out_width
#define conv2_in_ch conv1_out_ch
#define conv2_out_ch 256
#define conv2_out_width 6
#define conv2_stride 2
#define kernel_width 9
#define num_primary_caps 1152
#define dim_primary_caps 8
//primarycaps(dim_primary_caps, num_primary_caps)
#define dim_predic_vector 16
#define num_class 10
//prediction_vectors(num_primary_caps, num_class, dim_predic_vector)
#define num_iterations 1

#endif
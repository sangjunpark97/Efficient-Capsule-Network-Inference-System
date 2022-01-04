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

inline void prediction_vectors(char* input, char* weight_matrix, char* output) {
    char tmp;
    for (int l = 0; l < num_primary_caps; l++) {
        for (int k = 0; k < num_class; k++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                tmp = 0;
                for (int i = 0; i < dim_primary_caps; i++) {
                    tmp =   add_8(tmp,
                            mul_16(weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i],
                                input[i + l * dim_primary_caps]));
                }
                output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] = (tmp);
            }
        }
    }
}



inline void dynamic_routing(char* uhat, char* v_j) {
    int tmp = 0;
    int i, j, k, l;
    for (j = 0;j < num_class;j++) {
        for (k = 0;k < dim_predic_vector;k++) {
            for (l = 0;l < num_primary_caps;l++) {
                tmp += (int)uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
            }
            v_j[j * dim_predic_vector + k] = CLIP_8(tmp >> (4+7));
            tmp = 0;
        }
    }
}

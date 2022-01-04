
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#include "HLS/hls.h"
#include "HLS/ac_int.h"

#define fraction 4
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
 ihc::maxburst<512>,
 ihc::waitrequest<1>  > Master;

hls_avalon_slave_component component void convolution
(Master &input0,
 Master &kernel0,
 Master &kernel1,
 Master &output0
) {
    short int tmp;
    unsigned char i, j, p, l, m;
    unsigned short k;
    char (*input)[28] = (char(*)[28])input0;
    char (*conv1_kernel)[9][9] = (char(*)[9][9])kernel0;
    //char (*output)[20][20] = (char(*)[20][20])output0;
    char pri_input[16][20][20];
    for (i = 0; i < 20; i++) {
        for (j = 0; j < 20; j++) {
            for (k = 0; k < 16; k++) {
                tmp = 0;
                for (l = 0; l < 9; l++) {
                    for (m = 0; m < 9; m++) {
                        //tmp = add_16(tmp, mul_16(input0[28 * (i + l) + (j + m)], kernel0[k * 81 + l * 9 + m])); 
                        tmp = add_16(tmp, mul_16(input[i+l][j+m], conv1_kernel[k][l][m])); 
                    }                    
                }
                if(tmp < 0){
                    pri_input[k][i][j] = 0;
                    //pri_input0[k * 20 * 20 + i * 20 + j] = 0;
                }
                //else pri_input0[k * 20 * 20 + i * 20 + j] = CLIP_8(tmp);
                else pri_input[k][i][j] = CLIP_8(tmp);  
            }
        }
    }
    char (*conv2_kernel)[16][9][9] = (char(*)[16][9][9])kernel1;
    //char (*output)[6][6] = (char(*)[6][6])output0;
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 6; j++) {
            for (k = 0; k < 256; k++) {
                tmp = 0;
                for (p = 0; p < 16; p++) {
                    for (l = 0; l < 9; l++) {
                        for (m = 0; m < 9; m++) {
                            //tmp = add_16(tmp, mul_16(input0[i_wh * i_wh * p + i_wh * (i * S + l) + (j * S + m)], kernel[k * 81 * i_c + p * 81 + l * 9 + m]));
                            tmp = add_16(tmp, mul_16(pri_input[p][i*2+l][j*2+m], conv2_kernel[k][p][l][m]));
                        }
                    }
                }
                //output[k][i][j] = CLIP_8(tmp >> 7);   
                output0[k * 36 + i * 6 + j] = CLIP_8(tmp >> 7);
            }
        }
    }    
}
int main() {
    FILE* fp;
    char dataset[784];
    char conv1_kernel[16*9*9];  // ch      Ȯ   غ     
    char conv2_kernel[256*16*9*9];
    char conv2_output[256*6*6];
    
    ///////////////////////////////////////////
    fp = fopen("fx_conv_W.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_kernel, sizeof(char),16*9*9, fp);
    printf("conv1_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("fx_pri_W.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_kernel, sizeof(char), 256*16*9*9, fp);
    printf("conv2_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("fx_mnist_10000.bin", "rb"); if (fp == NULL)
    {
        printf("Cannot open file.\n");
        exit(1);
    }

    //for (int k = 0; k < 1; k++) {
        fread(dataset, sizeof(char), 784, fp);
        Master mm_a(dataset, sizeof(char)*784);
        Master mm_b(conv1_kernel, sizeof(char)*1296);
        Master mm_c(conv2_kernel, sizeof(char)*256*16*9*9);
        Master mm_d(conv2_output, sizeof(char)*256*6*6); 
        printf("\n");
        convolution(mm_a, mm_b,mm_c, mm_d);
        for (int i = 0;i < 1000;i++) {
            printf("%x ", conv2_output[i]);
        }
    //}
    fclose(fp);
    return 0;
}
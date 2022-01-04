
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/stdio.h"

#define fraction 4
#define CLIP_8(a) (a > 127? 127:(a < -128 ? -128: a))
#define CLIP_16(a) (a >  32767?  32767:(a < -32768 ? -32768: a))
#define mul_8(a,b) CLIP_8((a*b)>>fraction)
#define mul_16(a,b) CLIP_16(((short)a*(short)b)>>fraction)
#define add_8(a,b) CLIP_8(a + b)
#define add_16(a,b) CLIP_16(a + b)

typedef ihc::mm_master<char, ihc::dwidth<16>,
 ihc::awidth<32>,
 ihc::aspace<1>,
 ihc::latency<0>,
 ihc::maxburst<512>,
 ihc::align<2>,
 ihc::waitrequest<1>  > Master;


hls_avalon_slave_component component void prediction
(Master &input0,
 Master &kernel0,
 Master &output0
 ) {
    unsigned char i,j,k;
    unsigned short l;
    short tmp[16];
    char (*weight_matrix)[10][16][8] = (char(*)[10][16][8])kernel0;
    char (*input)[1152] = (char(*)[1152])input0;
    for (l = 0; l < 1152; l++) { // matmul
        for (k = 0; k < 10; k++) {
            for (j = 0; j < 16; j++) {
                tmp[j] = 0;
                for (i = 0; i < 8; i++) {
                    tmp[j] = add_16(tmp[j], mul_16(weight_matrix[l][k][j][i],input[i][l]));
                   //output0[j + 16 * k + 16 * 10 * l]  = add_16(output0[j + 16 * k + 16 * 10 * l], mul_16(kernel0[l * 16 * 8 * 10 + 16 * 8 * k + 8 * j + i], input0[1152*i + l]));
                }
                for(char x=0;x<16;x++){
                    output0[x + 16 * k + 16 * 10 * l] = CLIP_8(tmp[j]);
                }
            }
        }
    }
}
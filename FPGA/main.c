#include "system.h"
#include "io.h"
#include <stdio.h>
#include "fixedpoint.h"
#include <sys/alt_cache.h>
#include "sys/alt_alarm.h"
#include <altera_avalon_performance_counter.h>
#define PERFORMANCE_COUNTER_0_BASE 0x04000040
#define word_s 16
#define conv1_o_c 16
#define CLIP_8(a) (a > 127? 127:(a < -128 ? -128: a))
#define fraction 4
#define input_BASE       0x0
#define kernel0_BASE      0x500000
#define kernel1_BASE      0x1000000
#define output_BASE      0x3000000

#define weight_matrix_BASE     0x3700000
#define pred_output_BASE      0x3400000

#define mul_8(a,b) CLIP_8((a*b)>>fraction)
#define mul_16(a,b) CLIP_16((a*b)>>fraction)
#define add_8(a,b) CLIP_8(a + b)
#define add_16(a,b) CLIP_16(a + b)

char result_v[10 * 16];

void dynamic_routing(char* uhat, char* v_j) {
    int tmp = 0;
    int i, j, k, l;
     for (j = 0;j < 10;j++) {
        for (k = 0;k < 16;k++) {
            for (l = 0;l < 1152;l++) {
                tmp += uhat[10 * 16 * l + 16 * j + k];
            }
            v_j[j * 16 + k] = CLIP_8(tmp >> 11);
            tmp = 0;
        }
    }
}

int main() {
    volatile char* input_ptr = (char*)input_BASE;
    volatile char* kernel0_ptr = (char*)kernel0_BASE;
    volatile char* kernel1_ptr = (char*)kernel1_BASE;
    volatile char* output_ptr = (char*)output_BASE;
    volatile char* digit_W_ptr = (char*)weight_matrix_BASE;
    volatile char* pred_output_ptr = (char*)pred_output_BASE;
    volatile char* buffer = 0x2500000;

    FILE* fp;
    FILE* FI;
    FILE* FL;
    int correct = 0;
    int predict = 0;
    int max = 0;
    float accuracy;
    int predict_sum;
    int tmp_predict = 0;
    char LABEL;

    fp = fopen("/mnt/host/fx_conv_W.bin", "rb"); if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(kernel0_ptr, sizeof(char), conv1_o_c * 81, fp);

    printf("Conv1\n");
    fclose(fp);
    fp = fopen("/mnt/host/fx_pri_W.bin", "rb"); if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(kernel1_ptr, sizeof(char), 256*16 * 81, fp);

    printf("Conv2\n");
    fclose(fp);
    fp = fopen("/mnt/host/fx_digit_W.bin", "rb"); if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digit_W_ptr, sizeof(char), 1152* 10*16*8, fp);
    fclose(fp);
    FL = fopen("/mnt/host/label_char.bin", "rb"); if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }


    //int start_time, finish_time, total_time, start, finish, total;


    FI = fopen("/mnt/host/fx_mnist_10000.bin", "rb");
    if (FI == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    for (int k = 0; k < 10000;k++) {
        //fread(buffer, sizeof(char), 784, FI);
        fread(input_ptr, sizeof(char), 784, FI);
        fread(&LABEL, sizeof(char), 1, FL);
    	PERF_RESET(PERFORMANCE_COUNTER_0_BASE);
    	PERF_START_MEASURING(PERFORMANCE_COUNTER_0_BASE);
    	//PERF_BEGIN(PERFORMANCE_COUNTER_0_BASE, 0);
        /*for (int i = 0; i < 784;i++) {
            IOWR_8DIRECT(0x0, i, buffer[i]);
        }*/
        printf("Mnist\n");
        IOWR(0x04000018, 0, 1);
        IOWR(0x04000008, 0, 1);
        PERF_BEGIN(PERFORMANCE_COUNTER_0_BASE, 1);
        while (1) {
            if ((IORD(0x4000018, 0) >> 1) == 1)break;
        }
        PERF_END(PERFORMANCE_COUNTER_0_BASE, 1);

        IOWR(0x04000038, 0, 1);
        IOWR(0x04000028, 0, 1);
        PERF_BEGIN(PERFORMANCE_COUNTER_0_BASE, 2);

        while (1) {
            if ((IORD(0x4000038, 0) >> 1) == 1)break;
        }
        PERF_END(PERFORMANCE_COUNTER_0_BASE, 2);
        PERF_BEGIN(PERFORMANCE_COUNTER_0_BASE, 3);
        dynamic_routing(pred_output_ptr, result_v);
        PERF_END(PERFORMANCE_COUNTER_0_BASE, 3);
        printf("\n");
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < 16; j++) {
                        printf("%d, ", (result_v[j + i * 16]));
                    }
                    printf("\n");
                }
                max = 0;
                for (int i = 0; i < 10; i++) {
                    predict_sum = 0;
                    for (int j = 0; j < 16; j++) {
                        tmp_predict = (result_v[i * 16 + j] * result_v[i * 16 + j])>>fraction ;
                        predict_sum +=  tmp_predict;
                    }
                    if (predict_sum > max) {
                        max = predict_sum;
                        predict = i;
                    }
                }
                //PERF_END(PERFORMANCE_COUNTER_0_BASE, 0);
                PERF_STOP_MEASURING(PERFORMANCE_COUNTER_0_BASE);
                perf_print_formatted_report((void*)PERFORMANCE_COUNTER_0_BASE, alt_get_cpu_freq(), 3, "conv", "pred","DR");
                printf("(%d) pred : %d / ", k + 1, predict);
                if (predict == LABEL)  correct++;
                printf("target: %d / ", LABEL);
                accuracy = (float)correct / (k + 1);
                printf("accuracy : %.2f\n\n", accuracy * 100);
    }
    return 0;

}


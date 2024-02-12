/********************************************************************************
* Filename: config.h
* Date: $Sat Jan 20 14:25:05 2024
* Description: configuration file for the generated accelerator
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

// conv_0
#define CONV_0_K 3
#define CONV_0_IN_CH 3
#define CONV_0_IN_H 160
#define CONV_0_IN_W 320
#define CONV_0_OUT_CH 3
#define CONV_0_IN_BIT 8
#define CONV_0_W_BIT 8
#define CONV_0_INC_BIT 18
#define CONV_0_BIAS_BIT 32
#define CONV_0_OUT_BIT 8
#define CONV_0_SIMD 1
#define CONV_0_PE 1
#define CONV_0_L_SHIFT 8
#define CONV_0_ACTP 1
#define CONV_0_Kp 1
#define CONV_0_Np 2
#define CONV_0_GUARD_BIT 2
#define CONV_0_KPF 1

// conv_1
#define CONV_1_K 1
#define CONV_1_IN_CH 3
#define CONV_1_IN_H 160
#define CONV_1_IN_W 320
#define CONV_1_OUT_CH 48
#define CONV_1_IN_BIT 8
#define CONV_1_W_BIT 7
#define CONV_1_INC_BIT 20
#define CONV_1_BIAS_BIT 33
#define CONV_1_OUT_BIT 8
#define CONV_1_SIMD 3
#define CONV_1_PE 4
#define CONV_1_L_SHIFT 8
#define CONV_1_ACTP 4
#define CONV_1_Kp 1
#define CONV_1_Np 2
#define CONV_1_GUARD_BIT 2
#define CONV_1_KPF 1

// conv_2
#define CONV_2_K 3
#define CONV_2_IN_CH 48
#define CONV_2_IN_H 80
#define CONV_2_IN_W 160
#define CONV_2_OUT_CH 48
#define CONV_2_IN_BIT 8
#define CONV_2_W_BIT 6
#define CONV_2_INC_BIT 17
#define CONV_2_BIAS_BIT 26
#define CONV_2_OUT_BIT 3
#define CONV_2_SIMD 1
#define CONV_2_PE 4
#define CONV_2_L_SHIFT 8
#define CONV_2_ACTP 1
#define CONV_2_Kp 3
#define CONV_2_Np 2
#define CONV_2_GUARD_BIT 0
#define CONV_2_KPF 1

// conv_3
#define CONV_3_K 1
#define CONV_3_IN_CH 48
#define CONV_3_IN_H 80
#define CONV_3_IN_W 160
#define CONV_3_OUT_CH 96
#define CONV_3_IN_BIT 3
#define CONV_3_W_BIT 5
#define CONV_3_INC_BIT 17
#define CONV_3_BIAS_BIT 24
#define CONV_3_OUT_BIT 8
#define CONV_3_SIMD 8
#define CONV_3_PE 2
#define CONV_3_L_SHIFT 8
#define CONV_3_ACTP 2
#define CONV_3_Kp 2
#define CONV_3_Np 3
#define CONV_3_GUARD_BIT -1
#define CONV_3_KPF 1

// conv_4
#define CONV_4_K 3
#define CONV_4_IN_CH 96
#define CONV_4_IN_H 40
#define CONV_4_IN_W 80
#define CONV_4_OUT_CH 96
#define CONV_4_IN_BIT 8
#define CONV_4_W_BIT 6
#define CONV_4_INC_BIT 18
#define CONV_4_BIAS_BIT 28
#define CONV_4_OUT_BIT 7
#define CONV_4_SIMD 1
#define CONV_4_PE 2
#define CONV_4_L_SHIFT 8
#define CONV_4_ACTP 1
#define CONV_4_Kp 3
#define CONV_4_Np 2
#define CONV_4_GUARD_BIT 0
#define CONV_4_KPF 1

// conv_5
#define CONV_5_K 1
#define CONV_5_IN_CH 96
#define CONV_5_IN_H 40
#define CONV_5_IN_W 80
#define CONV_5_OUT_CH 192
#define CONV_5_IN_BIT 7
#define CONV_5_W_BIT 5
#define CONV_5_INC_BIT 17
#define CONV_5_BIAS_BIT 27
#define CONV_5_OUT_BIT 8
#define CONV_5_SIMD 6
#define CONV_5_PE 3
#define CONV_5_L_SHIFT 8
#define CONV_5_ACTP 1
#define CONV_5_Kp 1
#define CONV_5_Np 2
#define CONV_5_GUARD_BIT 7
#define CONV_5_KPF 1

// conv_6
#define CONV_6_K 3
#define CONV_6_IN_CH 192
#define CONV_6_IN_H 20
#define CONV_6_IN_W 40
#define CONV_6_OUT_CH 192
#define CONV_6_IN_BIT 8
#define CONV_6_W_BIT 8
#define CONV_6_INC_BIT 17
#define CONV_6_BIAS_BIT 29
#define CONV_6_OUT_BIT 5
#define CONV_6_SIMD 1
#define CONV_6_PE 1
#define CONV_6_L_SHIFT 8
#define CONV_6_ACTP 1
#define CONV_6_Kp 1
#define CONV_6_Np 2
#define CONV_6_GUARD_BIT 2
#define CONV_6_KPF 1

// conv_7
#define CONV_7_K 1
#define CONV_7_IN_CH 192
#define CONV_7_IN_H 20
#define CONV_7_IN_W 40
#define CONV_7_OUT_CH 384
#define CONV_7_IN_BIT 5
#define CONV_7_W_BIT 5
#define CONV_7_INC_BIT 15
#define CONV_7_BIAS_BIT 25
#define CONV_7_OUT_BIT 7
#define CONV_7_SIMD 6
#define CONV_7_PE 3
#define CONV_7_L_SHIFT 8
#define CONV_7_ACTP 1
#define CONV_7_Kp 2
#define CONV_7_Np 2
#define CONV_7_GUARD_BIT 0
#define CONV_7_KPF 1

// conv_8
#define CONV_8_K 3
#define CONV_8_IN_CH 384
#define CONV_8_IN_H 20
#define CONV_8_IN_W 40
#define CONV_8_OUT_CH 384
#define CONV_8_IN_BIT 7
#define CONV_8_W_BIT 8
#define CONV_8_INC_BIT 18
#define CONV_8_BIAS_BIT 28
#define CONV_8_OUT_BIT 5
#define CONV_8_SIMD 1
#define CONV_8_PE 2
#define CONV_8_L_SHIFT 8
#define CONV_8_ACTP 1
#define CONV_8_Kp 1
#define CONV_8_Np 2
#define CONV_8_GUARD_BIT 2
#define CONV_8_KPF 1

// conv_9
#define CONV_9_K 1
#define CONV_9_IN_CH 384
#define CONV_9_IN_H 20
#define CONV_9_IN_W 40
#define CONV_9_OUT_CH 512
#define CONV_9_IN_BIT 5
#define CONV_9_W_BIT 3
#define CONV_9_INC_BIT 13
#define CONV_9_BIAS_BIT 21
#define CONV_9_OUT_BIT 5
#define CONV_9_SIMD 12
#define CONV_9_PE 4
#define CONV_9_L_SHIFT 8
#define CONV_9_ACTP 1
#define CONV_9_Kp 2
#define CONV_9_Np 2
#define CONV_9_GUARD_BIT 3
#define CONV_9_KPF 1

// conv_10
#define CONV_10_K 3
#define CONV_10_IN_CH 512
#define CONV_10_IN_H 20
#define CONV_10_IN_W 40
#define CONV_10_OUT_CH 512
#define CONV_10_IN_BIT 5
#define CONV_10_W_BIT 6
#define CONV_10_INC_BIT 17
#define CONV_10_BIAS_BIT 24
#define CONV_10_OUT_BIT 5
#define CONV_10_SIMD 1
#define CONV_10_PE 4
#define CONV_10_L_SHIFT 8
#define CONV_10_ACTP 1
#define CONV_10_Kp 3
#define CONV_10_Np 2
#define CONV_10_GUARD_BIT 3
#define CONV_10_KPF 1

// conv_11
#define CONV_11_K 1
#define CONV_11_IN_CH 512
#define CONV_11_IN_H 20
#define CONV_11_IN_W 40
#define CONV_11_OUT_CH 96
#define CONV_11_IN_BIT 5
#define CONV_11_W_BIT 8
#define CONV_11_INC_BIT 15
#define CONV_11_BIAS_BIT 29
#define CONV_11_OUT_BIT 6
#define CONV_11_SIMD 8
#define CONV_11_PE 3
#define CONV_11_L_SHIFT 8
#define CONV_11_ACTP 1
#define CONV_11_Kp 1
#define CONV_11_Np 2
#define CONV_11_GUARD_BIT 8
#define CONV_11_KPF 1

// conv_12
#define CONV_12_K 1
#define CONV_12_IN_CH 96
#define CONV_12_IN_H 20
#define CONV_12_IN_W 40
#define CONV_12_OUT_CH 36
#define CONV_12_IN_BIT 6
#define CONV_12_W_BIT 8
#define CONV_12_OUT_BIT 32
#define CONV_12_SIMD 3
#define CONV_12_PE 2
#define CONV_12_ACTP 2
#define CONV_12_Kp 1
#define CONV_12_Np 2
#define CONV_12_GUARD_BIT 6
#define CONV_12_KPF 1

#endif

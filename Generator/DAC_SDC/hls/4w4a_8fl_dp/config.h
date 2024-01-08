/********************************************************************************
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

// conv_0
#define CONV_0_K 3
#define CONV_0_IN_CH 3
#define CONV_0_IN_H 160
#define CONV_0_IN_W 320
#define CONV_0_OUT_CH 16
#define CONV_0_IN_BIT 8
#define CONV_0_W_BIT 8
#define CONV_0_INC_BIT 15
#define CONV_0_BIAS_BIT 32
#define CONV_0_OUT_BIT 4
#define CONV_0_SIMD 3
#define CONV_0_PE 8
#define CONV_0_L_SHIFT 8
#define CONV_0_ACTP 8
#define CONV_0_Kp 2
#define CONV_0_Np 1
#define CONV_0_GUARD_BIT 3
#define CONV_0_IN_PE 3
#define CONV_0_KPF 3

// conv_1
#define CONV_1_K 3
#define CONV_1_IN_CH 16
#define CONV_1_IN_H 80
#define CONV_1_IN_W 160
#define CONV_1_OUT_CH 32
#define CONV_1_IN_BIT 4
#define CONV_1_W_BIT 4
#define CONV_1_INC_BIT 13
#define CONV_1_BIAS_BIT 22
#define CONV_1_OUT_BIT 4
#define CONV_1_SIMD 16
#define CONV_1_PE 4
#define CONV_1_L_SHIFT 8
#define CONV_1_ACTP 2
#define CONV_1_Kp 3
#define CONV_1_Np 2
#define CONV_1_GUARD_BIT 3
#define CONV_1_IN_PE 16
#define CONV_1_KPF 1

// conv_2
#define CONV_2_K 3
#define CONV_2_IN_CH 32
#define CONV_2_IN_H 40
#define CONV_2_IN_W 80
#define CONV_2_OUT_CH 64
#define CONV_2_IN_BIT 4
#define CONV_2_W_BIT 4
#define CONV_2_INC_BIT 13
#define CONV_2_BIAS_BIT 22
#define CONV_2_OUT_BIT 4
#define CONV_2_SIMD 32
#define CONV_2_PE 2
#define CONV_2_L_SHIFT 8
#define CONV_2_ACTP 1
#define CONV_2_Kp 3
#define CONV_2_Np 2
#define CONV_2_GUARD_BIT 3
#define CONV_2_IN_PE 4
#define CONV_2_KPF 1

// conv_3
#define CONV_3_K 3
#define CONV_3_IN_CH 64
#define CONV_3_IN_H 20
#define CONV_3_IN_W 40
#define CONV_3_OUT_CH 64
#define CONV_3_IN_BIT 4
#define CONV_3_W_BIT 4
#define CONV_3_INC_BIT 11
#define CONV_3_BIAS_BIT 21
#define CONV_3_OUT_BIT 4
#define CONV_3_SIMD 16
#define CONV_3_PE 2
#define CONV_3_L_SHIFT 8
#define CONV_3_ACTP 1
#define CONV_3_Kp 3
#define CONV_3_Np 2
#define CONV_3_GUARD_BIT 3
#define CONV_3_IN_PE 2
#define CONV_3_KPF 1

// conv_4
#define CONV_4_K 3
#define CONV_4_IN_CH 64
#define CONV_4_IN_H 10
#define CONV_4_IN_W 20
#define CONV_4_OUT_CH 64
#define CONV_4_IN_BIT 4
#define CONV_4_W_BIT 4
#define CONV_4_INC_BIT 11
#define CONV_4_BIAS_BIT 20
#define CONV_4_OUT_BIT 4
#define CONV_4_SIMD 4
#define CONV_4_PE 1
#define CONV_4_L_SHIFT 8
#define CONV_4_ACTP 1
#define CONV_4_Kp 3
#define CONV_4_Np 2
#define CONV_4_GUARD_BIT 3
#define CONV_4_IN_PE 2
#define CONV_4_KPF 3

// conv_5
#define CONV_5_K 3
#define CONV_5_IN_CH 64
#define CONV_5_IN_H 10
#define CONV_5_IN_W 20
#define CONV_5_OUT_CH 64
#define CONV_5_IN_BIT 4
#define CONV_5_W_BIT 4
#define CONV_5_INC_BIT 11
#define CONV_5_BIAS_BIT 20
#define CONV_5_OUT_BIT 4
#define CONV_5_SIMD 4
#define CONV_5_PE 1
#define CONV_5_L_SHIFT 8
#define CONV_5_ACTP 1
#define CONV_5_Kp 3
#define CONV_5_Np 2
#define CONV_5_GUARD_BIT 3
#define CONV_5_IN_PE 1
#define CONV_5_KPF 3

// conv_6
#define CONV_6_K 3
#define CONV_6_IN_CH 64
#define CONV_6_IN_H 10
#define CONV_6_IN_W 20
#define CONV_6_OUT_CH 64
#define CONV_6_IN_BIT 4
#define CONV_6_W_BIT 4
#define CONV_6_INC_BIT 12
#define CONV_6_BIAS_BIT 20
#define CONV_6_OUT_BIT 4
#define CONV_6_SIMD 4
#define CONV_6_PE 1
#define CONV_6_L_SHIFT 8
#define CONV_6_ACTP 1
#define CONV_6_Kp 3
#define CONV_6_Np 2
#define CONV_6_GUARD_BIT 3
#define CONV_6_IN_PE 1
#define CONV_6_KPF 3

// conv_7
#define CONV_7_K 3
#define CONV_7_IN_CH 64
#define CONV_7_IN_H 10
#define CONV_7_IN_W 20
#define CONV_7_OUT_CH 64
#define CONV_7_IN_BIT 4
#define CONV_7_W_BIT 4
#define CONV_7_INC_BIT 14
#define CONV_7_BIAS_BIT 23
#define CONV_7_OUT_BIT 4
#define CONV_7_SIMD 4
#define CONV_7_PE 1
#define CONV_7_L_SHIFT 8
#define CONV_7_ACTP 1
#define CONV_7_Kp 3
#define CONV_7_Np 2
#define CONV_7_GUARD_BIT 3
#define CONV_7_IN_PE 1
#define CONV_7_KPF 3

// conv_8
#define CONV_8_K 1
#define CONV_8_IN_CH 64
#define CONV_8_IN_H 10
#define CONV_8_IN_W 20
#define CONV_8_OUT_CH 36
#define CONV_8_IN_BIT 4
#define CONV_8_W_BIT 8
#define CONV_8_BIAS_BIT 15
#define CONV_8_OUT_BIT 32
#define CONV_8_SIMD 4
#define CONV_8_PE 2
#define CONV_8_ACTP 2
#define CONV_8_Kp 1
#define CONV_8_Np 2
#define CONV_8_GUARD_BIT 2
#define CONV_8_IN_PE 1
#define CONV_8_KPF 1

#endif

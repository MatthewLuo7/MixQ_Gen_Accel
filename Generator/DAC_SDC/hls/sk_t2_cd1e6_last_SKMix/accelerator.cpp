/********************************************************************************
* Filename: weights.hpp
* Date: $Mon Jan  8 17:55:12 2024
* Description: accelerator main function
********************************************************************************/
//#define DEBUG
//#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include <ap_int.h>
#include <iostream>
#include <stdint.h>

#include "config.h"
#include "weights.hpp"
#include "function.h"
#include "Opt_FP.hpp"
#include "Opt_FP_DW.hpp"
#include "Opt_FP_LUT.hpp"
#include "Opt_KP.hpp"
#include "Opt_KP_LUT.hpp"
#include "pool_reord.hpp"
#include "S2P_buffer.hpp"
#include "S2P_buffer_DW.hpp"
#include "stream_tools.h"



string output_path = "./debug_path/";

template <unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT, unsigned IN_NUM>
void input_quant(hls::stream<ap_uint<IN_BIT * IN_CH> > &in,
                 hls::stream<ap_uint<OUT_BIT * IN_CH> > &out,
                 const unsigned int reps){
  for(unsigned int i = 0; i < (IN_NUM * reps); i++){
#pragma HLS PIPELINE II = 1
    ap_uint<IN_BIT * IN_CH> indata = in.read();
    ap_uint<OUT_BIT * IN_CH> outdata;
    for(unsigned int j = 0; j < IN_CH; j++){
#pragma HLS unroll
      outdata(j * OUT_BIT + OUT_BIT - 1, j * OUT_BIT) = indata(j * IN_BIT + IN_BIT - 1, j * IN_BIT + IN_BIT - OUT_BIT);
    }
    out.write(outdata);
  }
}

void compute_pipeline(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
                 const unsigned int reps = 1) {
#pragma HLS DATAFLOW

  const unsigned int num_per_rep = 160 * 320 * 3 * 8 / 64;

  hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable = in_stream_extract depth = 256
  ExtractPixels<64, num_per_rep>(in, in_stream_extract, reps);

  hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable = in_stream0 depth = 256
  StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract,
                                                             in_stream0, reps);

  hls::stream<ap_uint<8 * CONV_0_IN_CH> > in_stream1("in_stream1");
#pragma HLS STREAM variable = in_stream1 depth = 512

  StreamingDataWidthConverter_Batch<64 * 3, 8 * CONV_0_IN_CH,
                                    num_per_rep / 3>(in_stream0, in_stream1,
                                                     reps);
#ifdef DEBUG
  cout << "in_stream1 size " << in_stream1.size() << endl;

#endif

  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IN_CH> > conv0_in("conv0_in");
#pragma HLS STREAM variable = conv0_in depth = 512
  const unsigned input_quant_num = 160 * 320 * 3 / CONV_0_IN_CH;
  input_quant<8, CONV_0_IN_CH, CONV_0_IN_BIT, input_quant_num>(in_stream1, conv0_in, reps);


/********************************************************************************Convolution 0********************************************************************************/

//--------------------Conv 0: Parameters--------------------
const unsigned CONV_0_IN_PE = 3;
stream<ap_uint<CONV_0_IN_PE * CONV_0_IN_BIT> > &conv_0_in = conv0_in;
const unsigned CONV_0_M_BIT = CONV_0_IN_BIT + CONV_0_W_BIT + 4;
const unsigned CONV_0_KPF_BIT = 0;
const unsigned CONV_0_CASCADE = 1;
const unsigned CONV_0_ROW_LEN = (CONV_0_IN_W + CONV_0_K - 1 - 1) / CONV_0_Np + 1;
const unsigned CONV_0_adW_BIT = 1;
const unsigned CONV_0_OCH_PF = CONV_0_PE;
const unsigned CONV_0_DEC_BW_NUM = CONV_0_IN_H * (CONV_0_OUT_CH / CONV_0_OCH_PF) * CONV_0_ROW_LEN;
const unsigned CONV_0_INC_BW_NUM = CONV_0_IN_H * (CONV_0_OUT_CH / CONV_0_OCH_PF) * CONV_0_IN_W * (CONV_0_OCH_PF / CONV_0_ACTP);
const unsigned CONV_0_W_Sep = 1;
const unsigned CONV_0_A_Sep = 1;
    
//--------------------Conv 0: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_0_PE * CONV_0_Np * CONV_0_KPF * CONV_0_IN_BIT> > conv_0_padding_out("conv_0_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_0_K, CONV_0_IN_H, CONV_0_IN_W, CONV_0_OUT_CH, CONV_0_OUT_CH / CONV_0_OCH_PF,
                              CONV_0_Np, CONV_0_IN_BIT, CONV_0_IN_PE, CONV_0_PE, 3,
                              9, 2, 8, 2, 8, 2, 8>(conv_0_in, conv_0_padding_out, reps);
    
//--------------------Conv 0: Computing Array--------------------
stream<ap_uint<CONV_0_Np * CONV_0_OCH_PF * CONV_0_M_BIT> > conv_0_array_out("conv_0_array_out");
FP_Array_DW<CONV_0_K, CONV_0_ROW_LEN, CONV_0_IN_H, CONV_0_OUT_CH,
            CONV_0_IN_BIT, CONV_0_W_BIT, CONV_0_KPF, CONV_0_PE, CONV_0_Kp,
            CONV_0_Np, CONV_0_CASCADE, CONV_0_GUARD_BIT, CONV_0_M_BIT, 
            CONV_0_KPF_BIT, CONV_0_adW_BIT, CONV_0_W_Sep, CONV_0_A_Sep,
            2, 2, 2, 4>(conv_0_padding_out, conv_0_w, conv_0_array_out, reps);
    
//--------------------Conv 0: Decrease Bit-width--------------------
stream<ap_uint<CONV_0_ACTP * CONV_0_M_BIT> > conv_0_dec_bw_out("conv_0_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_0_Np * CONV_0_OCH_PF * CONV_0_M_BIT, CONV_0_ACTP * CONV_0_M_BIT,
                                  CONV_0_DEC_BW_NUM>(conv_0_array_out, conv_0_dec_bw_out, reps);
    
//--------------------Conv 0: Activate and Trim--------------------
stream<ap_uint<CONV_0_ACTP * CONV_0_OUT_BIT> > conv_0_act_out("conv_0_act_out");
Activation_Trim<CONV_0_K, CONV_0_IN_W, CONV_0_ROW_LEN, CONV_0_IN_H, CONV_0_OUT_CH,
CONV_0_IN_BIT, CONV_0_OUT_BIT, CONV_0_W_BIT, CONV_0_INC_BIT, CONV_0_BIAS_BIT,
CONV_0_L_SHIFT, CONV_0_OCH_PF, CONV_0_ACTP, CONV_0_Np, CONV_0_M_BIT,
1, 9, 2>(conv_0_dec_bw_out, conv_0_inc, conv_0_bias, conv_0_act_out, reps);
    
//--------------------Conv 0: Increase Bit-width--------------------
stream<ap_uint<CONV_0_OCH_PF * CONV_0_OUT_BIT> > conv_0_layer_out("conv_0_layer_out");
#pragma HLS STREAM variable = conv_0_layer_out depth = 960
StreamingDataWidthConverter_Batch<CONV_0_ACTP * CONV_0_OUT_BIT, CONV_0_OCH_PF * CONV_0_OUT_BIT, CONV_0_INC_BW_NUM>(conv_0_act_out, conv_0_layer_out, reps);

#ifdef DEBUG
cout << "conv_0_layer_out size " << conv_0_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_0_IN_H, CONV_0_IN_W, CONV_0_OUT_CH, CONV_0_OCH_PF,
                                 CONV_1_IN_BIT>(conv_0_layer_out, output_path+"conv_0_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 1********************************************************************************/

//--------------------Conv 1: Parameters--------------------
const unsigned CONV_1_IN_PE = CONV_0_OCH_PF;
stream<ap_uint<CONV_1_IN_PE * CONV_1_IN_BIT> > &conv_1_in = conv_0_layer_out;
const unsigned CONV_1_M_BIT = CONV_1_IN_BIT + CONV_1_W_BIT + 2;
const unsigned CONV_1_SIMD_BIT = 2;
const unsigned CONV_1_CASCADE = 3;
const unsigned CONV_1_ROW_LEN = (CONV_1_IN_W + CONV_1_K - 1 - 1) / CONV_1_Np + 1;
const unsigned CONV_1_adW_BIT = 1;
const bool CONV_1_PatternFlag = false;
const unsigned CONV_1_OCH_PF = CONV_1_PE * CONV_1_Kp;
const unsigned CONV_1_DEC_BW_NUM = CONV_1_IN_H * (CONV_1_OUT_CH / CONV_1_OCH_PF) * CONV_1_ROW_LEN;
const unsigned CONV_1_INC_BW_NUM = CONV_1_IN_H * (CONV_1_OUT_CH / CONV_1_OCH_PF) * CONV_1_IN_W * (CONV_1_OCH_PF / CONV_1_ACTP);
const unsigned CONV_1_W_Sep = 1;
const unsigned CONV_1_A_Sep = 1;
    
//--------------------Conv 1: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_1_Np * CONV_1_SIMD * CONV_1_IN_BIT> > conv_1_padding_out("conv_1_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_1_K, CONV_1_IN_H, CONV_1_IN_W, CONV_1_IN_CH, CONV_1_OUT_CH / CONV_1_OCH_PF,
                             CONV_1_Np, CONV_1_IN_BIT, CONV_1_IN_PE, CONV_1_SIMD, 2,
                             9, 1, 2, 1, 9,
                             1, 1>(conv_1_in, conv_1_padding_out, reps);
    
//--------------------Conv 1: Computing Array--------------------
stream<ap_uint<CONV_1_Np * CONV_1_OCH_PF * CONV_1_M_BIT> > conv_1_array_out("conv_1_array_out");
KP_Array<CONV_1_K, CONV_1_ROW_LEN, CONV_1_IN_H, CONV_1_IN_CH, CONV_1_OUT_CH,
         CONV_1_IN_BIT, CONV_1_W_BIT, CONV_1_SIMD * CONV_1_KPF, CONV_1_PE, CONV_1_Kp,
         CONV_1_Np, CONV_1_CASCADE, CONV_1_GUARD_BIT, CONV_1_M_BIT, 
         CONV_1_SIMD_BIT, CONV_1_adW_BIT, CONV_1_W_Sep, CONV_1_A_Sep, CONV_1_PatternFlag,
         1, 2, 4>(conv_1_padding_out, conv_1_w, conv_1_array_out, reps);
    
//--------------------Conv 1: Decrease Bit-width--------------------
stream<ap_uint<CONV_1_ACTP * CONV_1_M_BIT> > conv_1_dec_bw_out("conv_1_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_1_Np * CONV_1_OCH_PF * CONV_1_M_BIT, CONV_1_ACTP * CONV_1_M_BIT,
                                  CONV_1_DEC_BW_NUM>(conv_1_array_out, conv_1_dec_bw_out, reps);
    
//--------------------Conv 1: Activate and Trim--------------------
stream<ap_uint<CONV_1_ACTP * CONV_1_OUT_BIT> > conv_1_act_out("conv_1_act_out");
Activation_Trim<CONV_1_K, CONV_1_IN_W, CONV_1_ROW_LEN, CONV_1_IN_H, CONV_1_OUT_CH,
CONV_1_IN_BIT, CONV_1_OUT_BIT, CONV_1_W_BIT, CONV_1_INC_BIT, CONV_1_BIAS_BIT,
CONV_1_L_SHIFT, CONV_1_OCH_PF, CONV_1_ACTP, CONV_1_Np, CONV_1_M_BIT,
1, 9, 4>(conv_1_dec_bw_out, conv_1_inc, conv_1_bias, conv_1_act_out, reps);
    
//--------------------Conv 1: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_1_OCH_PF * CONV_1_OUT_BIT> > conv_1_conv_out("conv_1_conv_out");
#pragma HLS STREAM variable = conv_1_conv_out depth = 7680
StreamingDataWidthConverter_Batch<CONV_1_ACTP * CONV_1_OUT_BIT, 2 * CONV_1_OCH_PF * CONV_1_OUT_BIT,
CONV_1_INC_BW_NUM>(conv_1_act_out, conv_1_conv_out, reps);

#ifdef DEBUG
cout << "conv_1_conv_out size " << conv_1_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_1_IN_H, CONV_1_IN_W, CONV_1_OUT_CH, CONV_1_OCH_PF,
                                    CONV_2_IN_BIT>(conv_1_conv_out, output_path+"conv_1_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_1_OCH_PF * CONV_1_OUT_BIT> > conv_1_layer_out("conv_1_layer_out");
#pragma HLS STREAM variable = conv_1_layer_out depth = 3840
max_pool2x2<CONV_1_IN_H, CONV_1_IN_W, CONV_1_OUT_CH, CONV_1_OUT_BIT,
            CONV_1_OCH_PF>(conv_1_conv_out, conv_1_layer_out, reps);
#ifdef DEBUG
cout << "conv_1_pool_out size " << conv_1_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_1_IN_H / 2, CONV_1_IN_W / 2,
                           CONV_1_OUT_CH, CONV_1_OCH_PF, CONV_1_OUT_BIT>(conv_1_layer_out, output_path+"conv_1_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 2********************************************************************************/

//--------------------Conv 2: Parameters--------------------
const unsigned CONV_2_IN_PE = CONV_1_OCH_PF;
stream<ap_uint<CONV_2_IN_PE * CONV_2_IN_BIT> > &conv_2_in = conv_1_layer_out;
const unsigned CONV_2_M_BIT = CONV_2_IN_BIT + CONV_2_W_BIT + 4;
const unsigned CONV_2_KPF_BIT = 0;
const unsigned CONV_2_CASCADE = 1;
const unsigned CONV_2_ROW_LEN = (CONV_2_IN_W + CONV_2_K - 1 - 1) / CONV_2_Np + 1;
const unsigned CONV_2_adW_BIT = 1;
const unsigned CONV_2_OCH_PF = CONV_2_PE;
const unsigned CONV_2_DEC_BW_NUM = CONV_2_IN_H * (CONV_2_OUT_CH / CONV_2_OCH_PF) * CONV_2_ROW_LEN;
const unsigned CONV_2_INC_BW_NUM = CONV_2_IN_H * (CONV_2_OUT_CH / CONV_2_OCH_PF) * CONV_2_IN_W * (CONV_2_OCH_PF / CONV_2_ACTP);
const unsigned CONV_2_W_Sep = 1;
const unsigned CONV_2_A_Sep = 1;
    
//--------------------Conv 2: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_2_PE * CONV_2_Np * CONV_2_KPF * CONV_2_IN_BIT> > conv_2_padding_out("conv_2_padding_out");
DW_reshape_buffer_PE_INPE_S2P<CONV_2_K, CONV_2_IN_H, CONV_2_IN_W, CONV_2_OUT_CH, CONV_2_OUT_CH / CONV_2_OCH_PF,
                             CONV_2_Np, CONV_2_IN_BIT, CONV_2_IN_PE, CONV_2_PE,
                             3, 8, 2, 1, 10, 7, 2, 10>(conv_2_in, conv_2_padding_out, reps);
    
//--------------------Conv 2: Computing Array--------------------
stream<ap_uint<CONV_2_Np * CONV_2_OCH_PF * CONV_2_M_BIT> > conv_2_array_out("conv_2_array_out");
FP_Array_DW<CONV_2_K, CONV_2_ROW_LEN, CONV_2_IN_H, CONV_2_OUT_CH,
            CONV_2_IN_BIT, CONV_2_W_BIT, CONV_2_KPF, CONV_2_PE, CONV_2_Kp,
            CONV_2_Np, CONV_2_CASCADE, CONV_2_GUARD_BIT, CONV_2_M_BIT, 
            CONV_2_KPF_BIT, CONV_2_adW_BIT, CONV_2_W_Sep, CONV_2_A_Sep,
            2, 2, 2, 6>(conv_2_padding_out, conv_2_w, conv_2_array_out, reps);
    
//--------------------Conv 2: Decrease Bit-width--------------------
stream<ap_uint<CONV_2_ACTP * CONV_2_M_BIT> > conv_2_dec_bw_out("conv_2_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_2_Np * CONV_2_OCH_PF * CONV_2_M_BIT, CONV_2_ACTP * CONV_2_M_BIT,
                                  CONV_2_DEC_BW_NUM>(conv_2_array_out, conv_2_dec_bw_out, reps);
    
//--------------------Conv 2: Activate and Trim--------------------
stream<ap_uint<CONV_2_ACTP * CONV_2_OUT_BIT> > conv_2_act_out("conv_2_act_out");
Activation_Trim<CONV_2_K, CONV_2_IN_W, CONV_2_ROW_LEN, CONV_2_IN_H, CONV_2_OUT_CH,
CONV_2_IN_BIT, CONV_2_OUT_BIT, CONV_2_W_BIT, CONV_2_INC_BIT, CONV_2_BIAS_BIT,
CONV_2_L_SHIFT, CONV_2_OCH_PF, CONV_2_ACTP, CONV_2_Np, CONV_2_M_BIT,
3, 8, 6>(conv_2_dec_bw_out, conv_2_inc, conv_2_bias, conv_2_act_out, reps);
    
//--------------------Conv 2: Increase Bit-width--------------------
stream<ap_uint<CONV_2_OCH_PF * CONV_2_OUT_BIT> > conv_2_layer_out("conv_2_layer_out");
#pragma HLS STREAM variable = conv_2_layer_out depth = 1920
StreamingDataWidthConverter_Batch<CONV_2_ACTP * CONV_2_OUT_BIT, CONV_2_OCH_PF * CONV_2_OUT_BIT, CONV_2_INC_BW_NUM>(conv_2_act_out, conv_2_layer_out, reps);

#ifdef DEBUG
cout << "conv_2_layer_out size " << conv_2_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_2_IN_H, CONV_2_IN_W, CONV_2_OUT_CH, CONV_2_OCH_PF,
                                 CONV_3_IN_BIT>(conv_2_layer_out, output_path+"conv_2_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 3********************************************************************************/

//--------------------Conv 3: Parameters--------------------
const unsigned CONV_3_IN_PE = CONV_2_OCH_PF;
stream<ap_uint<CONV_3_IN_PE * CONV_3_IN_BIT> > &conv_3_in = conv_2_layer_out;
const unsigned CONV_3_M_BIT = CONV_3_IN_BIT + CONV_3_W_BIT + 6;
const unsigned CONV_3_SIMD_BIT = 3;
const unsigned CONV_3_CASCADE = 1;
const unsigned CONV_3_ROW_LEN = (CONV_3_IN_W + CONV_3_K - 1 - 1) / CONV_3_Np + 1;
const unsigned CONV_3_adW_BIT = 1;
const bool CONV_3_PatternFlag = false;
const unsigned CONV_3_OCH_PF = CONV_3_PE * CONV_3_Kp;
const unsigned CONV_3_DEC_BW_NUM = CONV_3_IN_H * (CONV_3_OUT_CH / CONV_3_OCH_PF) * CONV_3_ROW_LEN;
const unsigned CONV_3_INC_BW_NUM = CONV_3_IN_H * (CONV_3_OUT_CH / CONV_3_OCH_PF) * CONV_3_IN_W * (CONV_3_OCH_PF / CONV_3_ACTP);
const unsigned CONV_3_W_Sep = 1;
const unsigned CONV_3_A_Sep = 1;
    
//--------------------Conv 3: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_3_Np * CONV_3_SIMD * CONV_3_IN_BIT> > conv_3_padding_out("conv_3_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_3_K, CONV_3_IN_H, CONV_3_IN_W, CONV_3_IN_CH, CONV_3_OUT_CH / CONV_3_OCH_PF,
                             CONV_3_Np, CONV_3_IN_BIT, CONV_3_IN_PE, CONV_3_SIMD, 2,
                             8, 2, 2, 3, 9,
                             1, 3>(conv_3_in, conv_3_padding_out, reps);
    
//--------------------Conv 3: Computing Array--------------------
stream<ap_uint<CONV_3_Np * CONV_3_OCH_PF * CONV_3_M_BIT> > conv_3_array_out("conv_3_array_out");
KP_Array<CONV_3_K, CONV_3_ROW_LEN, CONV_3_IN_H, CONV_3_IN_CH, CONV_3_OUT_CH,
         CONV_3_IN_BIT, CONV_3_W_BIT, CONV_3_SIMD * CONV_3_KPF, CONV_3_PE, CONV_3_Kp,
         CONV_3_Np, CONV_3_CASCADE, CONV_3_GUARD_BIT, CONV_3_M_BIT, 
         CONV_3_SIMD_BIT, CONV_3_adW_BIT, CONV_3_W_Sep, CONV_3_A_Sep, CONV_3_PatternFlag,
         1, 3, 8>(conv_3_padding_out, conv_3_w, conv_3_array_out, reps);
    
//--------------------Conv 3: Decrease Bit-width--------------------
stream<ap_uint<CONV_3_ACTP * CONV_3_M_BIT> > conv_3_dec_bw_out("conv_3_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_3_Np * CONV_3_OCH_PF * CONV_3_M_BIT, CONV_3_ACTP * CONV_3_M_BIT,
                                  CONV_3_DEC_BW_NUM>(conv_3_array_out, conv_3_dec_bw_out, reps);
    
//--------------------Conv 3: Activate and Trim--------------------
stream<ap_uint<CONV_3_ACTP * CONV_3_OUT_BIT> > conv_3_act_out("conv_3_act_out");
Activation_Trim<CONV_3_K, CONV_3_IN_W, CONV_3_ROW_LEN, CONV_3_IN_H, CONV_3_OUT_CH,
CONV_3_IN_BIT, CONV_3_OUT_BIT, CONV_3_W_BIT, CONV_3_INC_BIT, CONV_3_BIAS_BIT,
CONV_3_L_SHIFT, CONV_3_OCH_PF, CONV_3_ACTP, CONV_3_Np, CONV_3_M_BIT,
2, 8, 6>(conv_3_dec_bw_out, conv_3_inc, conv_3_bias, conv_3_act_out, reps);
    
//--------------------Conv 3: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_3_OCH_PF * CONV_3_OUT_BIT> > conv_3_conv_out("conv_3_conv_out");
#pragma HLS STREAM variable = conv_3_conv_out depth = 7680
StreamingDataWidthConverter_Batch<CONV_3_ACTP * CONV_3_OUT_BIT, 2 * CONV_3_OCH_PF * CONV_3_OUT_BIT,
CONV_3_INC_BW_NUM>(conv_3_act_out, conv_3_conv_out, reps);

#ifdef DEBUG
cout << "conv_3_conv_out size " << conv_3_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_3_IN_H, CONV_3_IN_W, CONV_3_OUT_CH, CONV_3_OCH_PF,
                                    CONV_4_IN_BIT>(conv_3_conv_out, output_path+"conv_3_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_3_OCH_PF * CONV_3_OUT_BIT> > conv_3_layer_out("conv_3_layer_out");
#pragma HLS STREAM variable = conv_3_layer_out depth = 3840
max_pool2x2<CONV_3_IN_H, CONV_3_IN_W, CONV_3_OUT_CH, CONV_3_OUT_BIT,
            CONV_3_OCH_PF>(conv_3_conv_out, conv_3_layer_out, reps);
#ifdef DEBUG
cout << "conv_3_pool_out size " << conv_3_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_3_IN_H / 2, CONV_3_IN_W / 2,
                           CONV_3_OUT_CH, CONV_3_OCH_PF, CONV_3_OUT_BIT>(conv_3_layer_out, output_path+"conv_3_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 4********************************************************************************/

//--------------------Conv 4: Parameters--------------------
const unsigned CONV_4_IN_PE = CONV_3_OCH_PF;
stream<ap_uint<CONV_4_IN_PE * CONV_4_IN_BIT> > &conv_4_in = conv_3_layer_out;
const unsigned CONV_4_M_BIT = CONV_4_IN_BIT + CONV_4_W_BIT + 4;
const unsigned CONV_4_KPF_BIT = 0;
const unsigned CONV_4_CASCADE = 1;
const unsigned CONV_4_ROW_LEN = (CONV_4_IN_W + CONV_4_K - 1 - 1) / CONV_4_Np + 1;
const unsigned CONV_4_adW_BIT = 1;
const unsigned CONV_4_OCH_PF = CONV_4_PE;
const unsigned CONV_4_DEC_BW_NUM = CONV_4_IN_H * (CONV_4_OUT_CH / CONV_4_OCH_PF) * CONV_4_ROW_LEN;
const unsigned CONV_4_INC_BW_NUM = CONV_4_IN_H * (CONV_4_OUT_CH / CONV_4_OCH_PF) * CONV_4_IN_W * (CONV_4_OCH_PF / CONV_4_ACTP);
const unsigned CONV_4_W_Sep = 1;
const unsigned CONV_4_A_Sep = 1;
    
//--------------------Conv 4: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_4_PE * CONV_4_Np * CONV_4_KPF * CONV_4_IN_BIT> > conv_4_padding_out("conv_4_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_4_K, CONV_4_IN_H, CONV_4_IN_W, CONV_4_OUT_CH, CONV_4_OUT_CH / CONV_4_OCH_PF,
                              CONV_4_Np, CONV_4_IN_BIT, CONV_4_IN_PE, CONV_4_PE, 3,
                              7, 2, 10, 2, 10, 2, 6>(conv_4_in, conv_4_padding_out, reps);
    
//--------------------Conv 4: Computing Array--------------------
stream<ap_uint<CONV_4_Np * CONV_4_OCH_PF * CONV_4_M_BIT> > conv_4_array_out("conv_4_array_out");
FP_Array_DW<CONV_4_K, CONV_4_ROW_LEN, CONV_4_IN_H, CONV_4_OUT_CH,
            CONV_4_IN_BIT, CONV_4_W_BIT, CONV_4_KPF, CONV_4_PE, CONV_4_Kp,
            CONV_4_Np, CONV_4_CASCADE, CONV_4_GUARD_BIT, CONV_4_M_BIT, 
            CONV_4_KPF_BIT, CONV_4_adW_BIT, CONV_4_W_Sep, CONV_4_A_Sep,
            2, 2, 2, 8>(conv_4_padding_out, conv_4_w, conv_4_array_out, reps);
    
//--------------------Conv 4: Decrease Bit-width--------------------
stream<ap_uint<CONV_4_ACTP * CONV_4_M_BIT> > conv_4_dec_bw_out("conv_4_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_4_Np * CONV_4_OCH_PF * CONV_4_M_BIT, CONV_4_ACTP * CONV_4_M_BIT,
                                  CONV_4_DEC_BW_NUM>(conv_4_array_out, conv_4_dec_bw_out, reps);
    
//--------------------Conv 4: Activate and Trim--------------------
stream<ap_uint<CONV_4_ACTP * CONV_4_OUT_BIT> > conv_4_act_out("conv_4_act_out");
Activation_Trim<CONV_4_K, CONV_4_IN_W, CONV_4_ROW_LEN, CONV_4_IN_H, CONV_4_OUT_CH,
CONV_4_IN_BIT, CONV_4_OUT_BIT, CONV_4_W_BIT, CONV_4_INC_BIT, CONV_4_BIAS_BIT,
CONV_4_L_SHIFT, CONV_4_OCH_PF, CONV_4_ACTP, CONV_4_Np, CONV_4_M_BIT,
2, 7, 7>(conv_4_dec_bw_out, conv_4_inc, conv_4_bias, conv_4_act_out, reps);
    
//--------------------Conv 4: Increase Bit-width--------------------
stream<ap_uint<CONV_4_OCH_PF * CONV_4_OUT_BIT> > conv_4_layer_out("conv_4_layer_out");
#pragma HLS STREAM variable = conv_4_layer_out depth = 3840
StreamingDataWidthConverter_Batch<CONV_4_ACTP * CONV_4_OUT_BIT, CONV_4_OCH_PF * CONV_4_OUT_BIT, CONV_4_INC_BW_NUM>(conv_4_act_out, conv_4_layer_out, reps);

#ifdef DEBUG
cout << "conv_4_layer_out size " << conv_4_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_4_IN_H, CONV_4_IN_W, CONV_4_OUT_CH, CONV_4_OCH_PF,
                                 CONV_5_IN_BIT>(conv_4_layer_out, output_path+"conv_4_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 5********************************************************************************/

//--------------------Conv 5: Parameters--------------------
const unsigned CONV_5_IN_PE = CONV_4_OCH_PF;
stream<ap_uint<CONV_5_IN_PE * CONV_5_IN_BIT> > &conv_5_in = conv_4_layer_out;
const unsigned CONV_5_M_BIT = CONV_5_IN_BIT + CONV_5_W_BIT + 7;
const unsigned CONV_5_SIMD_BIT = 3;
const unsigned CONV_5_CASCADE = 1;
const unsigned CONV_5_ROW_LEN = (CONV_5_IN_W + CONV_5_K - 1 - 1) / CONV_5_Np + 1;
const unsigned CONV_5_adW_BIT = 1;
const bool CONV_5_PatternFlag = false;
const unsigned CONV_5_OCH_PF = CONV_5_PE * CONV_5_Kp;
const unsigned CONV_5_DEC_BW_NUM = CONV_5_IN_H * (CONV_5_OUT_CH / CONV_5_OCH_PF) * CONV_5_ROW_LEN;
const unsigned CONV_5_INC_BW_NUM = CONV_5_IN_H * (CONV_5_OUT_CH / CONV_5_OCH_PF) * CONV_5_IN_W * (CONV_5_OCH_PF / CONV_5_ACTP);
const unsigned CONV_5_W_Sep = 1;
const unsigned CONV_5_A_Sep = 1;
    
//--------------------Conv 5: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_5_Np * CONV_5_SIMD * CONV_5_IN_BIT> > conv_5_padding_out("conv_5_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_5_K, CONV_5_IN_H, CONV_5_IN_W, CONV_5_IN_CH, CONV_5_OUT_CH / CONV_5_OCH_PF,
                             CONV_5_Np, CONV_5_IN_BIT, CONV_5_IN_PE, CONV_5_SIMD, 2,
                             7, 2, 2, 5, 10,
                             1, 5>(conv_5_in, conv_5_padding_out, reps);
    
//--------------------Conv 5: Computing Array--------------------
stream<ap_uint<CONV_5_Np * CONV_5_OCH_PF * CONV_5_M_BIT> > conv_5_array_out("conv_5_array_out");
KP_Array<CONV_5_K, CONV_5_ROW_LEN, CONV_5_IN_H, CONV_5_IN_CH, CONV_5_OUT_CH,
         CONV_5_IN_BIT, CONV_5_W_BIT, CONV_5_SIMD * CONV_5_KPF, CONV_5_PE, CONV_5_Kp,
         CONV_5_Np, CONV_5_CASCADE, CONV_5_GUARD_BIT, CONV_5_M_BIT, 
         CONV_5_SIMD_BIT, CONV_5_adW_BIT, CONV_5_W_Sep, CONV_5_A_Sep, CONV_5_PatternFlag,
         1, 5, 10>(conv_5_padding_out, conv_5_w, conv_5_array_out, reps);
    
//--------------------Conv 5: Decrease Bit-width--------------------
stream<ap_uint<CONV_5_ACTP * CONV_5_M_BIT> > conv_5_dec_bw_out("conv_5_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_5_Np * CONV_5_OCH_PF * CONV_5_M_BIT, CONV_5_ACTP * CONV_5_M_BIT,
                                  CONV_5_DEC_BW_NUM>(conv_5_array_out, conv_5_dec_bw_out, reps);
    
//--------------------Conv 5: Activate and Trim--------------------
stream<ap_uint<CONV_5_ACTP * CONV_5_OUT_BIT> > conv_5_act_out("conv_5_act_out");
Activation_Trim<CONV_5_K, CONV_5_IN_W, CONV_5_ROW_LEN, CONV_5_IN_H, CONV_5_OUT_CH,
CONV_5_IN_BIT, CONV_5_OUT_BIT, CONV_5_W_BIT, CONV_5_INC_BIT, CONV_5_BIAS_BIT,
CONV_5_L_SHIFT, CONV_5_OCH_PF, CONV_5_ACTP, CONV_5_Np, CONV_5_M_BIT,
3, 7, 8>(conv_5_dec_bw_out, conv_5_inc, conv_5_bias, conv_5_act_out, reps);
    
//--------------------Conv 5: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_5_OCH_PF * CONV_5_OUT_BIT> > conv_5_conv_out("conv_5_conv_out");
#pragma HLS STREAM variable = conv_5_conv_out depth = 5120
StreamingDataWidthConverter_Batch<CONV_5_ACTP * CONV_5_OUT_BIT, 2 * CONV_5_OCH_PF * CONV_5_OUT_BIT,
CONV_5_INC_BW_NUM>(conv_5_act_out, conv_5_conv_out, reps);

#ifdef DEBUG
cout << "conv_5_conv_out size " << conv_5_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_5_IN_H, CONV_5_IN_W, CONV_5_OUT_CH, CONV_5_OCH_PF,
                                    CONV_6_IN_BIT>(conv_5_conv_out, output_path+"conv_5_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_5_OCH_PF * CONV_5_OUT_BIT> > conv_5_layer_out("conv_5_layer_out");
#pragma HLS STREAM variable = conv_5_layer_out depth = 2560
max_pool2x2<CONV_5_IN_H, CONV_5_IN_W, CONV_5_OUT_CH, CONV_5_OUT_BIT,
            CONV_5_OCH_PF>(conv_5_conv_out, conv_5_layer_out, reps);
#ifdef DEBUG
cout << "conv_5_pool_out size " << conv_5_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_5_IN_H / 2, CONV_5_IN_W / 2,
                           CONV_5_OUT_CH, CONV_5_OCH_PF, CONV_5_OUT_BIT>(conv_5_layer_out, output_path+"conv_5_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 6********************************************************************************/

//--------------------Conv 6: Parameters--------------------
const unsigned CONV_6_IN_PE = CONV_5_OCH_PF;
stream<ap_uint<CONV_6_IN_PE * CONV_6_IN_BIT> > &conv_6_in = conv_5_layer_out;
const unsigned CONV_6_M_BIT = CONV_6_IN_BIT + CONV_6_W_BIT + 4;
const unsigned CONV_6_KPF_BIT = 0;
const unsigned CONV_6_CASCADE = 1;
const unsigned CONV_6_ROW_LEN = (CONV_6_IN_W + CONV_6_K - 1 - 1) / CONV_6_Np + 1;
const unsigned CONV_6_adW_BIT = 1;
const unsigned CONV_6_OCH_PF = CONV_6_PE;
const unsigned CONV_6_DEC_BW_NUM = CONV_6_IN_H * (CONV_6_OUT_CH / CONV_6_OCH_PF) * CONV_6_ROW_LEN;
const unsigned CONV_6_INC_BW_NUM = CONV_6_IN_H * (CONV_6_OUT_CH / CONV_6_OCH_PF) * CONV_6_IN_W * (CONV_6_OCH_PF / CONV_6_ACTP);
const unsigned CONV_6_W_Sep = 1;
const unsigned CONV_6_A_Sep = 1;
    
//--------------------Conv 6: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_6_PE * CONV_6_Np * CONV_6_KPF * CONV_6_IN_BIT> > conv_6_padding_out("conv_6_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_6_K, CONV_6_IN_H, CONV_6_IN_W, CONV_6_OUT_CH, CONV_6_OUT_CH / CONV_6_OCH_PF,
                              CONV_6_Np, CONV_6_IN_BIT, CONV_6_IN_PE, CONV_6_PE, 3,
                              6, 2, 10, 2, 10, 3, 5>(conv_6_in, conv_6_padding_out, reps);
    
//--------------------Conv 6: Computing Array--------------------
stream<ap_uint<CONV_6_Np * CONV_6_OCH_PF * CONV_6_M_BIT> > conv_6_array_out("conv_6_array_out");
FP_Array_DW<CONV_6_K, CONV_6_ROW_LEN, CONV_6_IN_H, CONV_6_OUT_CH,
            CONV_6_IN_BIT, CONV_6_W_BIT, CONV_6_KPF, CONV_6_PE, CONV_6_Kp,
            CONV_6_Np, CONV_6_CASCADE, CONV_6_GUARD_BIT, CONV_6_M_BIT, 
            CONV_6_KPF_BIT, CONV_6_adW_BIT, CONV_6_W_Sep, CONV_6_A_Sep,
            2, 2, 2, 10>(conv_6_padding_out, conv_6_w, conv_6_array_out, reps);
    
//--------------------Conv 6: Decrease Bit-width--------------------
stream<ap_uint<CONV_6_ACTP * CONV_6_M_BIT> > conv_6_dec_bw_out("conv_6_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_6_Np * CONV_6_OCH_PF * CONV_6_M_BIT, CONV_6_ACTP * CONV_6_M_BIT,
                                  CONV_6_DEC_BW_NUM>(conv_6_array_out, conv_6_dec_bw_out, reps);
    
//--------------------Conv 6: Activate and Trim--------------------
stream<ap_uint<CONV_6_ACTP * CONV_6_OUT_BIT> > conv_6_act_out("conv_6_act_out");
Activation_Trim<CONV_6_K, CONV_6_IN_W, CONV_6_ROW_LEN, CONV_6_IN_H, CONV_6_OUT_CH,
CONV_6_IN_BIT, CONV_6_OUT_BIT, CONV_6_W_BIT, CONV_6_INC_BIT, CONV_6_BIAS_BIT,
CONV_6_L_SHIFT, CONV_6_OCH_PF, CONV_6_ACTP, CONV_6_Np, CONV_6_M_BIT,
1, 6, 8>(conv_6_dec_bw_out, conv_6_inc, conv_6_bias, conv_6_act_out, reps);
    
//--------------------Conv 6: Increase Bit-width--------------------
stream<ap_uint<CONV_6_OCH_PF * CONV_6_OUT_BIT> > conv_6_layer_out("conv_6_layer_out");
#pragma HLS STREAM variable = conv_6_layer_out depth = 7680
StreamingDataWidthConverter_Batch<CONV_6_ACTP * CONV_6_OUT_BIT, CONV_6_OCH_PF * CONV_6_OUT_BIT, CONV_6_INC_BW_NUM>(conv_6_act_out, conv_6_layer_out, reps);

#ifdef DEBUG
cout << "conv_6_layer_out size " << conv_6_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_6_IN_H, CONV_6_IN_W, CONV_6_OUT_CH, CONV_6_OCH_PF,
                                 CONV_7_IN_BIT>(conv_6_layer_out, output_path+"conv_6_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 7********************************************************************************/

//--------------------Conv 7: Parameters--------------------
const unsigned CONV_7_IN_PE = CONV_6_OCH_PF;
stream<ap_uint<CONV_7_IN_PE * CONV_7_IN_BIT> > &conv_7_in = conv_6_layer_out;
const unsigned CONV_7_M_BIT = CONV_7_IN_BIT + CONV_7_W_BIT + 8;
const unsigned CONV_7_SIMD_BIT = 3;
const unsigned CONV_7_CASCADE = 2;
const unsigned CONV_7_ROW_LEN = (CONV_7_IN_W + CONV_7_K - 1 - 1) / CONV_7_Np + 1;
const unsigned CONV_7_adW_BIT = 1;
const bool CONV_7_PatternFlag = false;
const unsigned CONV_7_OCH_PF = CONV_7_PE * CONV_7_Kp;
const unsigned CONV_7_DEC_BW_NUM = CONV_7_IN_H * (CONV_7_OUT_CH / CONV_7_OCH_PF) * CONV_7_ROW_LEN;
const unsigned CONV_7_INC_BW_NUM = CONV_7_IN_H * (CONV_7_OUT_CH / CONV_7_OCH_PF) * CONV_7_IN_W * (CONV_7_OCH_PF / CONV_7_ACTP);
const unsigned CONV_7_W_Sep = 1;
const unsigned CONV_7_A_Sep = 1;
    
//--------------------Conv 7: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_7_Np * CONV_7_SIMD * CONV_7_IN_BIT> > conv_7_padding_out("conv_7_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_7_K, CONV_7_IN_H, CONV_7_IN_W, CONV_7_IN_CH, CONV_7_OUT_CH / CONV_7_OCH_PF,
                             CONV_7_Np, CONV_7_IN_BIT, CONV_7_IN_PE, CONV_7_SIMD, 2,
                             6, 2, 3, 6, 10,
                             1, 6>(conv_7_in, conv_7_padding_out, reps);
    
//--------------------Conv 7: Computing Array--------------------
stream<ap_uint<CONV_7_Np * CONV_7_OCH_PF * CONV_7_M_BIT> > conv_7_array_out("conv_7_array_out");
KP_Array<CONV_7_K, CONV_7_ROW_LEN, CONV_7_IN_H, CONV_7_IN_CH, CONV_7_OUT_CH,
         CONV_7_IN_BIT, CONV_7_W_BIT, CONV_7_SIMD * CONV_7_KPF, CONV_7_PE, CONV_7_Kp,
         CONV_7_Np, CONV_7_CASCADE, CONV_7_GUARD_BIT, CONV_7_M_BIT, 
         CONV_7_SIMD_BIT, CONV_7_adW_BIT, CONV_7_W_Sep, CONV_7_A_Sep, CONV_7_PatternFlag,
         1, 6, 12>(conv_7_padding_out, conv_7_w, conv_7_array_out, reps);
    
//--------------------Conv 7: Decrease Bit-width--------------------
stream<ap_uint<CONV_7_ACTP * CONV_7_M_BIT> > conv_7_dec_bw_out("conv_7_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_7_Np * CONV_7_OCH_PF * CONV_7_M_BIT, CONV_7_ACTP * CONV_7_M_BIT,
                                  CONV_7_DEC_BW_NUM>(conv_7_array_out, conv_7_dec_bw_out, reps);
    
//--------------------Conv 7: Activate and Trim--------------------
stream<ap_uint<CONV_7_ACTP * CONV_7_OUT_BIT> > conv_7_act_out("conv_7_act_out");
Activation_Trim<CONV_7_K, CONV_7_IN_W, CONV_7_ROW_LEN, CONV_7_IN_H, CONV_7_OUT_CH,
CONV_7_IN_BIT, CONV_7_OUT_BIT, CONV_7_W_BIT, CONV_7_INC_BIT, CONV_7_BIAS_BIT,
CONV_7_L_SHIFT, CONV_7_OCH_PF, CONV_7_ACTP, CONV_7_Np, CONV_7_M_BIT,
3, 6, 9>(conv_7_dec_bw_out, conv_7_inc, conv_7_bias, conv_7_act_out, reps);
    
//--------------------Conv 7: Increase Bit-width--------------------
stream<ap_uint<CONV_7_OCH_PF * CONV_7_OUT_BIT> > conv_7_layer_out("conv_7_layer_out");
#pragma HLS STREAM variable = conv_7_layer_out depth = 5120
StreamingDataWidthConverter_Batch<CONV_7_ACTP * CONV_7_OUT_BIT, CONV_7_OCH_PF * CONV_7_OUT_BIT, CONV_7_INC_BW_NUM>(conv_7_act_out, conv_7_layer_out, reps);

#ifdef DEBUG
cout << "conv_7_layer_out size " << conv_7_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_7_IN_H, CONV_7_IN_W, CONV_7_OUT_CH, CONV_7_OCH_PF,
                                 CONV_8_IN_BIT>(conv_7_layer_out, output_path+"conv_7_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 8********************************************************************************/

//--------------------Conv 8: Parameters--------------------
const unsigned CONV_8_IN_PE = CONV_7_OCH_PF;
stream<ap_uint<CONV_8_IN_PE * CONV_8_IN_BIT> > &conv_8_in = conv_7_layer_out;
const unsigned CONV_8_M_BIT = CONV_8_IN_BIT + CONV_8_W_BIT + 4;
const unsigned CONV_8_KPF_BIT = 0;
const unsigned CONV_8_CASCADE = 1;
const unsigned CONV_8_ROW_LEN = (CONV_8_IN_W + CONV_8_K - 1 - 1) / CONV_8_Np + 1;
const unsigned CONV_8_adW_BIT = 1;
const unsigned CONV_8_OCH_PF = CONV_8_PE;
const unsigned CONV_8_DEC_BW_NUM = CONV_8_IN_H * (CONV_8_OUT_CH / CONV_8_OCH_PF) * CONV_8_ROW_LEN;
const unsigned CONV_8_INC_BW_NUM = CONV_8_IN_H * (CONV_8_OUT_CH / CONV_8_OCH_PF) * CONV_8_IN_W * (CONV_8_OCH_PF / CONV_8_ACTP);
const unsigned CONV_8_W_Sep = 1;
const unsigned CONV_8_A_Sep = 1;
    
//--------------------Conv 8: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_8_PE * CONV_8_Np * CONV_8_KPF * CONV_8_IN_BIT> > conv_8_padding_out("conv_8_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_8_K, CONV_8_IN_H, CONV_8_IN_W, CONV_8_OUT_CH, CONV_8_OUT_CH / CONV_8_OCH_PF,
                              CONV_8_Np, CONV_8_IN_BIT, CONV_8_IN_PE, CONV_8_PE, 3,
                              6, 2, 11, 2, 11, 2, 5>(conv_8_in, conv_8_padding_out, reps);
    
//--------------------Conv 8: Computing Array--------------------
stream<ap_uint<CONV_8_Np * CONV_8_OCH_PF * CONV_8_M_BIT> > conv_8_array_out("conv_8_array_out");
FP_Array_DW<CONV_8_K, CONV_8_ROW_LEN, CONV_8_IN_H, CONV_8_OUT_CH,
            CONV_8_IN_BIT, CONV_8_W_BIT, CONV_8_KPF, CONV_8_PE, CONV_8_Kp,
            CONV_8_Np, CONV_8_CASCADE, CONV_8_GUARD_BIT, CONV_8_M_BIT, 
            CONV_8_KPF_BIT, CONV_8_adW_BIT, CONV_8_W_Sep, CONV_8_A_Sep,
            2, 2, 2, 10>(conv_8_padding_out, conv_8_w, conv_8_array_out, reps);
    
//--------------------Conv 8: Decrease Bit-width--------------------
stream<ap_uint<CONV_8_ACTP * CONV_8_M_BIT> > conv_8_dec_bw_out("conv_8_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_8_Np * CONV_8_OCH_PF * CONV_8_M_BIT, CONV_8_ACTP * CONV_8_M_BIT,
                                  CONV_8_DEC_BW_NUM>(conv_8_array_out, conv_8_dec_bw_out, reps);
    
//--------------------Conv 8: Activate and Trim--------------------
stream<ap_uint<CONV_8_ACTP * CONV_8_OUT_BIT> > conv_8_act_out("conv_8_act_out");
Activation_Trim<CONV_8_K, CONV_8_IN_W, CONV_8_ROW_LEN, CONV_8_IN_H, CONV_8_OUT_CH,
CONV_8_IN_BIT, CONV_8_OUT_BIT, CONV_8_W_BIT, CONV_8_INC_BIT, CONV_8_BIAS_BIT,
CONV_8_L_SHIFT, CONV_8_OCH_PF, CONV_8_ACTP, CONV_8_Np, CONV_8_M_BIT,
2, 6, 9>(conv_8_dec_bw_out, conv_8_inc, conv_8_bias, conv_8_act_out, reps);
    
//--------------------Conv 8: Increase Bit-width--------------------
stream<ap_uint<CONV_8_OCH_PF * CONV_8_OUT_BIT> > conv_8_layer_out("conv_8_layer_out");
#pragma HLS STREAM variable = conv_8_layer_out depth = 7680
StreamingDataWidthConverter_Batch<CONV_8_ACTP * CONV_8_OUT_BIT, CONV_8_OCH_PF * CONV_8_OUT_BIT, CONV_8_INC_BW_NUM>(conv_8_act_out, conv_8_layer_out, reps);

#ifdef DEBUG
cout << "conv_8_layer_out size " << conv_8_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_8_IN_H, CONV_8_IN_W, CONV_8_OUT_CH, CONV_8_OCH_PF,
                                 CONV_9_IN_BIT>(conv_8_layer_out, output_path+"conv_8_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 9********************************************************************************/

//--------------------Conv 9: Parameters--------------------
const unsigned CONV_9_IN_PE = CONV_8_OCH_PF;
stream<ap_uint<CONV_9_IN_PE * CONV_9_IN_BIT> > &conv_9_in = conv_8_layer_out;
const unsigned CONV_9_M_BIT = CONV_9_IN_BIT + CONV_9_W_BIT + 9;
const unsigned CONV_9_SIMD_BIT = 4;
const unsigned CONV_9_CASCADE = 6;
const unsigned CONV_9_ROW_LEN = (CONV_9_IN_W + CONV_9_K - 1 - 1) / CONV_9_Np + 1;
const unsigned CONV_9_adW_BIT = 1;
const bool CONV_9_PatternFlag = false;
const unsigned CONV_9_OCH_PF = CONV_9_PE * CONV_9_Kp;
const unsigned CONV_9_DEC_BW_NUM = CONV_9_IN_H * (CONV_9_OUT_CH / CONV_9_OCH_PF) * CONV_9_ROW_LEN;
const unsigned CONV_9_INC_BW_NUM = CONV_9_IN_H * (CONV_9_OUT_CH / CONV_9_OCH_PF) * CONV_9_IN_W * (CONV_9_OCH_PF / CONV_9_ACTP);
const unsigned CONV_9_W_Sep = 1;
const unsigned CONV_9_A_Sep = 1;
    
//--------------------Conv 9: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_9_Np * CONV_9_SIMD * CONV_9_IN_BIT> > conv_9_padding_out("conv_9_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_9_K, CONV_9_IN_H, CONV_9_IN_W, CONV_9_IN_CH, CONV_9_OUT_CH / CONV_9_OCH_PF,
                             CONV_9_Np, CONV_9_IN_BIT, CONV_9_IN_PE, CONV_9_SIMD, 2,
                             6, 2, 3, 6, 10,
                             1, 6>(conv_9_in, conv_9_padding_out, reps);
    
//--------------------Conv 9: Computing Array--------------------
stream<ap_uint<CONV_9_Np * CONV_9_OCH_PF * CONV_9_M_BIT> > conv_9_array_out("conv_9_array_out");
KP_Array<CONV_9_K, CONV_9_ROW_LEN, CONV_9_IN_H, CONV_9_IN_CH, CONV_9_OUT_CH,
         CONV_9_IN_BIT, CONV_9_W_BIT, CONV_9_SIMD * CONV_9_KPF, CONV_9_PE, CONV_9_Kp,
         CONV_9_Np, CONV_9_CASCADE, CONV_9_GUARD_BIT, CONV_9_M_BIT, 
         CONV_9_SIMD_BIT, CONV_9_adW_BIT, CONV_9_W_Sep, CONV_9_A_Sep, CONV_9_PatternFlag,
         1, 6, 12>(conv_9_padding_out, conv_9_w, conv_9_array_out, reps);
    
//--------------------Conv 9: Decrease Bit-width--------------------
stream<ap_uint<CONV_9_ACTP * CONV_9_M_BIT> > conv_9_dec_bw_out("conv_9_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_9_Np * CONV_9_OCH_PF * CONV_9_M_BIT, CONV_9_ACTP * CONV_9_M_BIT,
                                  CONV_9_DEC_BW_NUM>(conv_9_array_out, conv_9_dec_bw_out, reps);
    
//--------------------Conv 9: Activate and Trim--------------------
stream<ap_uint<CONV_9_ACTP * CONV_9_OUT_BIT> > conv_9_act_out("conv_9_act_out");
Activation_Trim<CONV_9_K, CONV_9_IN_W, CONV_9_ROW_LEN, CONV_9_IN_H, CONV_9_OUT_CH,
CONV_9_IN_BIT, CONV_9_OUT_BIT, CONV_9_W_BIT, CONV_9_INC_BIT, CONV_9_BIAS_BIT,
CONV_9_L_SHIFT, CONV_9_OCH_PF, CONV_9_ACTP, CONV_9_Np, CONV_9_M_BIT,
4, 6, 10>(conv_9_dec_bw_out, conv_9_inc, conv_9_bias, conv_9_act_out, reps);
    
//--------------------Conv 9: Increase Bit-width--------------------
stream<ap_uint<CONV_9_OCH_PF * CONV_9_OUT_BIT> > conv_9_layer_out("conv_9_layer_out");
#pragma HLS STREAM variable = conv_9_layer_out depth = 5120
StreamingDataWidthConverter_Batch<CONV_9_ACTP * CONV_9_OUT_BIT, CONV_9_OCH_PF * CONV_9_OUT_BIT, CONV_9_INC_BW_NUM>(conv_9_act_out, conv_9_layer_out, reps);

#ifdef DEBUG
cout << "conv_9_layer_out size " << conv_9_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_9_IN_H, CONV_9_IN_W, CONV_9_OUT_CH, CONV_9_OCH_PF,
                                 CONV_10_IN_BIT>(conv_9_layer_out, output_path+"conv_9_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 10********************************************************************************/

//--------------------Conv 10: Parameters--------------------
const unsigned CONV_10_IN_PE = CONV_9_OCH_PF;
stream<ap_uint<CONV_10_IN_PE * CONV_10_IN_BIT> > &conv_10_in = conv_9_layer_out;
const unsigned CONV_10_M_BIT = CONV_10_IN_BIT + CONV_10_W_BIT + 4;
const unsigned CONV_10_KPF_BIT = 0;
const unsigned CONV_10_CASCADE = 1;
const unsigned CONV_10_ROW_LEN = (CONV_10_IN_W + CONV_10_K - 1 - 1) / CONV_10_Np + 1;
const unsigned CONV_10_adW_BIT = 1;
const unsigned CONV_10_OCH_PF = CONV_10_PE;
const unsigned CONV_10_DEC_BW_NUM = CONV_10_IN_H * (CONV_10_OUT_CH / CONV_10_OCH_PF) * CONV_10_ROW_LEN;
const unsigned CONV_10_INC_BW_NUM = CONV_10_IN_H * (CONV_10_OUT_CH / CONV_10_OCH_PF) * CONV_10_IN_W * (CONV_10_OCH_PF / CONV_10_ACTP);
const unsigned CONV_10_W_Sep = 1;
const unsigned CONV_10_A_Sep = 1;
    
//--------------------Conv 10: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_10_PE * CONV_10_Np * CONV_10_KPF * CONV_10_IN_BIT> > conv_10_padding_out("conv_10_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_10_K, CONV_10_IN_H, CONV_10_IN_W, CONV_10_OUT_CH, CONV_10_OUT_CH / CONV_10_OCH_PF,
                              CONV_10_Np, CONV_10_IN_BIT, CONV_10_IN_PE, CONV_10_PE, 3,
                              6, 2, 11, 2, 11, 2, 5>(conv_10_in, conv_10_padding_out, reps);
    
//--------------------Conv 10: Computing Array--------------------
stream<ap_uint<CONV_10_Np * CONV_10_OCH_PF * CONV_10_M_BIT> > conv_10_array_out("conv_10_array_out");
FP_Array_DW<CONV_10_K, CONV_10_ROW_LEN, CONV_10_IN_H, CONV_10_OUT_CH,
            CONV_10_IN_BIT, CONV_10_W_BIT, CONV_10_KPF, CONV_10_PE, CONV_10_Kp,
            CONV_10_Np, CONV_10_CASCADE, CONV_10_GUARD_BIT, CONV_10_M_BIT, 
            CONV_10_KPF_BIT, CONV_10_adW_BIT, CONV_10_W_Sep, CONV_10_A_Sep,
            2, 2, 2, 9>(conv_10_padding_out, conv_10_w, conv_10_array_out, reps);
    
//--------------------Conv 10: Decrease Bit-width--------------------
stream<ap_uint<CONV_10_ACTP * CONV_10_M_BIT> > conv_10_dec_bw_out("conv_10_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_10_Np * CONV_10_OCH_PF * CONV_10_M_BIT, CONV_10_ACTP * CONV_10_M_BIT,
                                  CONV_10_DEC_BW_NUM>(conv_10_array_out, conv_10_dec_bw_out, reps);
    
//--------------------Conv 10: Activate and Trim--------------------
stream<ap_uint<CONV_10_ACTP * CONV_10_OUT_BIT> > conv_10_act_out("conv_10_act_out");
Activation_Trim<CONV_10_K, CONV_10_IN_W, CONV_10_ROW_LEN, CONV_10_IN_H, CONV_10_OUT_CH,
CONV_10_IN_BIT, CONV_10_OUT_BIT, CONV_10_W_BIT, CONV_10_INC_BIT, CONV_10_BIAS_BIT,
CONV_10_L_SHIFT, CONV_10_OCH_PF, CONV_10_ACTP, CONV_10_Np, CONV_10_M_BIT,
3, 6, 10>(conv_10_dec_bw_out, conv_10_inc, conv_10_bias, conv_10_act_out, reps);
    
//--------------------Conv 10: Increase Bit-width--------------------
stream<ap_uint<CONV_10_OCH_PF * CONV_10_OUT_BIT> > conv_10_layer_out("conv_10_layer_out");
#pragma HLS STREAM variable = conv_10_layer_out depth = 5120
StreamingDataWidthConverter_Batch<CONV_10_ACTP * CONV_10_OUT_BIT, CONV_10_OCH_PF * CONV_10_OUT_BIT, CONV_10_INC_BW_NUM>(conv_10_act_out, conv_10_layer_out, reps);

#ifdef DEBUG
cout << "conv_10_layer_out size " << conv_10_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_10_IN_H, CONV_10_IN_W, CONV_10_OUT_CH, CONV_10_OCH_PF,
                                 CONV_11_IN_BIT>(conv_10_layer_out, output_path+"conv_10_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 11********************************************************************************/

//--------------------Conv 11: Parameters--------------------
const unsigned CONV_11_IN_PE = CONV_10_OCH_PF;
stream<ap_uint<CONV_11_IN_PE * CONV_11_IN_BIT> > &conv_11_in = conv_10_layer_out;
const unsigned CONV_11_M_BIT = CONV_11_IN_BIT + CONV_11_W_BIT + 9;
const unsigned CONV_11_SIMD_BIT = 3;
const unsigned CONV_11_CASCADE = 8;
const unsigned CONV_11_ROW_LEN = (CONV_11_IN_W + CONV_11_K - 1 - 1) / CONV_11_Np + 1;
const unsigned CONV_11_adW_BIT = 1;
const bool CONV_11_PatternFlag = false;
const unsigned CONV_11_OCH_PF = CONV_11_PE * CONV_11_Kp;
const unsigned CONV_11_DEC_BW_NUM = CONV_11_IN_H * (CONV_11_OUT_CH / CONV_11_OCH_PF) * CONV_11_ROW_LEN;
const unsigned CONV_11_INC_BW_NUM = CONV_11_IN_H * (CONV_11_OUT_CH / CONV_11_OCH_PF) * CONV_11_IN_W * (CONV_11_OCH_PF / CONV_11_ACTP);
const unsigned CONV_11_W_Sep = 1;
const unsigned CONV_11_A_Sep = 1;
    
//--------------------Conv 11: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_11_Np * CONV_11_SIMD * CONV_11_IN_BIT> > conv_11_padding_out("conv_11_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_11_K, CONV_11_IN_H, CONV_11_IN_W, CONV_11_IN_CH, CONV_11_OUT_CH / CONV_11_OCH_PF,
                             CONV_11_Np, CONV_11_IN_BIT, CONV_11_IN_PE, CONV_11_SIMD, 2,
                             6, 2, 2, 7, 11,
                             1, 7>(conv_11_in, conv_11_padding_out, reps);
    
//--------------------Conv 11: Computing Array--------------------
stream<ap_uint<CONV_11_Np * CONV_11_OCH_PF * CONV_11_M_BIT> > conv_11_array_out("conv_11_array_out");
KP_Array<CONV_11_K, CONV_11_ROW_LEN, CONV_11_IN_H, CONV_11_IN_CH, CONV_11_OUT_CH,
         CONV_11_IN_BIT, CONV_11_W_BIT, CONV_11_SIMD * CONV_11_KPF, CONV_11_PE, CONV_11_Kp,
         CONV_11_Np, CONV_11_CASCADE, CONV_11_GUARD_BIT, CONV_11_M_BIT, 
         CONV_11_SIMD_BIT, CONV_11_adW_BIT, CONV_11_W_Sep, CONV_11_A_Sep, CONV_11_PatternFlag,
         1, 7, 12>(conv_11_padding_out, conv_11_w, conv_11_array_out, reps);
    
//--------------------Conv 11: Decrease Bit-width--------------------
stream<ap_uint<CONV_11_ACTP * CONV_11_M_BIT> > conv_11_dec_bw_out("conv_11_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_11_Np * CONV_11_OCH_PF * CONV_11_M_BIT, CONV_11_ACTP * CONV_11_M_BIT,
                                  CONV_11_DEC_BW_NUM>(conv_11_array_out, conv_11_dec_bw_out, reps);
    
//--------------------Conv 11: Activate and Trim--------------------
stream<ap_uint<CONV_11_ACTP * CONV_11_OUT_BIT> > conv_11_act_out("conv_11_act_out");
Activation_Trim<CONV_11_K, CONV_11_IN_W, CONV_11_ROW_LEN, CONV_11_IN_H, CONV_11_OUT_CH,
CONV_11_IN_BIT, CONV_11_OUT_BIT, CONV_11_W_BIT, CONV_11_INC_BIT, CONV_11_BIAS_BIT,
CONV_11_L_SHIFT, CONV_11_OCH_PF, CONV_11_ACTP, CONV_11_Np, CONV_11_M_BIT,
2, 6, 7>(conv_11_dec_bw_out, conv_11_inc, conv_11_bias, conv_11_act_out, reps);
    
//--------------------Conv 11: Increase Bit-width--------------------
stream<ap_uint<CONV_11_OCH_PF * CONV_11_OUT_BIT> > conv_11_layer_out("conv_11_layer_out");
#pragma HLS STREAM variable = conv_11_layer_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_11_ACTP * CONV_11_OUT_BIT, CONV_11_OCH_PF * CONV_11_OUT_BIT, CONV_11_INC_BW_NUM>(conv_11_act_out, conv_11_layer_out, reps);

#ifdef DEBUG
cout << "conv_11_layer_out size " << conv_11_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_11_IN_H, CONV_11_IN_W, CONV_11_OUT_CH, CONV_11_OCH_PF,
                                 CONV_12_IN_BIT>(conv_11_layer_out, output_path+"conv_11_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 12********************************************************************************/

//--------------------Conv 12: Parameters--------------------
const unsigned CONV_12_IN_PE = CONV_11_OCH_PF;
stream<ap_uint<CONV_12_IN_PE * CONV_12_IN_BIT> > &conv_12_in = conv_11_layer_out;
const unsigned CONV_12_M_BIT = 32;
const unsigned CONV_12_SIMD_BIT = 2;
const unsigned CONV_12_CASCADE = 1;
const unsigned CONV_12_ROW_LEN = (CONV_12_IN_W + CONV_12_K - 1 - 1) / CONV_12_Np + 1;
const unsigned CONV_12_adW_BIT = 1;
const bool CONV_12_PatternFlag = false;
const unsigned CONV_12_OCH_PF = CONV_12_PE * CONV_12_Kp;
const unsigned CONV_12_DEC_BW_NUM = CONV_12_IN_H * (CONV_12_OUT_CH / CONV_12_OCH_PF) * CONV_12_ROW_LEN;
const unsigned CONV_12_INC_BW_NUM = CONV_12_IN_H * (CONV_12_OUT_CH / CONV_12_OCH_PF) * CONV_12_IN_W * (CONV_12_OCH_PF / CONV_12_ACTP);
const unsigned CONV_12_W_Sep = 1;
const unsigned CONV_12_A_Sep = 1;
    
//--------------------Conv 12: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_12_Np * CONV_12_SIMD * CONV_12_IN_BIT> > conv_12_padding_out("conv_12_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_12_K, CONV_12_IN_H, CONV_12_IN_W, CONV_12_IN_CH, CONV_12_OUT_CH / CONV_12_OCH_PF,
                             CONV_12_Np, CONV_12_IN_BIT, CONV_12_IN_PE, CONV_12_SIMD, 2,
                             6, 1, 1, 6, 11,
                             1, 6>(conv_12_in, conv_12_padding_out, reps);
    
//--------------------Conv 12: Computing Array--------------------
stream<ap_uint<CONV_12_Np * CONV_12_OCH_PF * CONV_12_M_BIT> > conv_12_array_out("conv_12_array_out");
KP_Array<CONV_12_K, CONV_12_ROW_LEN, CONV_12_IN_H, CONV_12_IN_CH, CONV_12_OUT_CH,
         CONV_12_IN_BIT, CONV_12_W_BIT, CONV_12_SIMD * CONV_12_KPF, CONV_12_PE, CONV_12_Kp,
         CONV_12_Np, CONV_12_CASCADE, CONV_12_GUARD_BIT, CONV_12_M_BIT, 
         CONV_12_SIMD_BIT, CONV_12_adW_BIT, CONV_12_W_Sep, CONV_12_A_Sep, CONV_12_PatternFlag,
         1, 6, 10>(conv_12_padding_out, conv_12_w, conv_12_array_out, reps);
    
//--------------------Conv 12: Decrease Bit-width--------------------
stream<ap_uint<CONV_12_ACTP * CONV_12_M_BIT> > conv_12_dec_bw_out("conv_12_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_12_Np * CONV_12_OCH_PF * CONV_12_M_BIT, CONV_12_ACTP * CONV_12_M_BIT,
                                  CONV_12_DEC_BW_NUM>(conv_12_array_out, conv_12_dec_bw_out, reps);

stream<ap_uint<CONV_12_ACTP * CONV_12_OUT_BIT> > conv_12_act_out("conv_12_act_out");
Trim<CONV_12_K, CONV_12_IN_W, CONV_12_ROW_LEN, CONV_12_IN_H, CONV_12_OUT_CH,
CONV_12_OUT_BIT,  CONV_12_OCH_PF, CONV_12_ACTP, CONV_12_Np,
1, 6, 5>(conv_12_dec_bw_out, conv_12_act_out, reps);

//-------------------- Add Last --------------------
AddLast<CONV_12_IN_H * CONV_12_IN_W * CONV_12_OUT_CH / 2>(conv_12_act_out, out, reps);
}

void sky_net(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
               const unsigned reps) {

#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_8_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_8_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_9_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_9_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_9_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_10_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_10_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_10_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_11_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_11_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_11_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_12_w complete dim = 1



  compute_pipeline(in, out, reps);
}


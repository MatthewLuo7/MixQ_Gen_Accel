//#define DEBUG
//#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include "config.h"
#include "function.h"
#include "pool_reord.hpp"
#include "stream_tools.h"
#include "weights.hpp"
#include <ap_int.h>
#include <iostream>
#include <stdint.h>
#include "S2P_buffer.hpp"
#include "Opt_FP.hpp"
#include "Opt_FP_LUT.hpp"
#include "Opt_KP.hpp"
#include "Opt_KP_LUT.hpp"

string output_path = "E:/Projects/DeepBurning_MixQ/Accel_test_2_4w4a/5_gen_2/debug_output";

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
stream<ap_uint<CONV_0_IN_PE * CONV_0_IN_BIT> > &conv_0_in = conv0_in;
const unsigned CONV_0_M_BIT = CONV_0_IN_BIT + CONV_0_W_BIT + 5;
const unsigned CONV_0_SIMD_BIT = 4;
const unsigned CONV_0_ROW_LEN = (CONV_0_IN_W + CONV_0_K - 1 - 1) / CONV_0_Np + 1;
const unsigned CONV_0_OCH_PF = CONV_0_PE * CONV_0_Kp;
const unsigned CONV_0_DEC_BW_NUM = CONV_0_IN_H * (CONV_0_OUT_CH / CONV_0_OCH_PF) * CONV_0_ROW_LEN;
const unsigned CONV_0_INC_BW_NUM = CONV_0_IN_H * (CONV_0_OUT_CH / CONV_0_OCH_PF) * CONV_0_IN_W * (CONV_0_OCH_PF / CONV_0_ACTP);
    
//--------------------Conv 0: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_0_Np * CONV_0_K * CONV_0_SIMD * CONV_0_IN_BIT> > conv_0_padding_out("conv_0_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_0_K, CONV_0_IN_H, CONV_0_IN_W, CONV_0_IN_CH, CONV_0_OUT_CH / CONV_0_OCH_PF,
                             CONV_0_Np, CONV_0_IN_BIT, CONV_0_IN_PE, CONV_0_SIMD, 3,
                             9, 2, 2, 2, 8, 2>(conv_0_in, conv_0_padding_out, reps);
    
//--------------------Conv 0: Computing Array--------------------
stream<ap_uint<CONV_0_Np * CONV_0_OCH_PF * CONV_0_M_BIT> > conv_0_array_out("conv_0_array_out");
KP_Array_lut<CONV_0_K, CONV_0_ROW_LEN, CONV_0_IN_H, CONV_0_IN_CH, CONV_0_OUT_CH,
             CONV_0_IN_BIT, CONV_0_W_BIT, CONV_0_SIMD * CONV_0_KPF, CONV_0_PE, CONV_0_Kp,
             CONV_0_Np, CONV_0_M_BIT, CONV_0_SIMD_BIT, 2, 2, 3>(conv_0_padding_out, conv_0_w, conv_0_array_out, reps);
    
//--------------------Conv 0: Decrease Bit-width--------------------
stream<ap_uint<CONV_0_ACTP * CONV_0_M_BIT> > conv_0_dec_bw_out("conv_0_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_0_Np * CONV_0_OCH_PF * CONV_0_M_BIT, CONV_0_ACTP * CONV_0_M_BIT,
                                  CONV_0_DEC_BW_NUM>(conv_0_array_out, conv_0_dec_bw_out, reps);
    
//--------------------Conv 0: Activate and Trim--------------------
stream<ap_uint<CONV_0_ACTP * CONV_0_OUT_BIT> > conv_0_act_out("conv_0_act_out");
Activation_Trim<CONV_0_K, CONV_0_IN_W, CONV_0_ROW_LEN, CONV_0_IN_H, CONV_0_OUT_CH,
CONV_0_IN_BIT, CONV_0_OUT_BIT, CONV_0_W_BIT, CONV_0_INC_BIT, CONV_0_BIAS_BIT,
CONV_0_L_SHIFT, CONV_0_OCH_PF, CONV_0_ACTP, CONV_0_Np, CONV_0_M_BIT,
2, 9, 2>(conv_0_dec_bw_out, conv_0_inc, conv_0_bias, conv_0_act_out, reps);
    
//--------------------Conv 0: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_0_OCH_PF * CONV_0_OUT_BIT> > conv_0_conv_out("conv_0_conv_out");
#pragma HLS STREAM variable = conv_0_conv_out depth = 640
StreamingDataWidthConverter_Batch<CONV_0_ACTP * CONV_0_OUT_BIT, 2 * CONV_0_OCH_PF * CONV_0_OUT_BIT,
CONV_0_INC_BW_NUM>(conv_0_act_out, conv_0_conv_out, reps);

#ifdef DEBUG
cout << "conv_0_conv_out size " << conv_0_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_0_IN_H, CONV_0_IN_W, CONV_0_OUT_CH, CONV_0_OCH_PF,
                                    CONV_1_IN_BIT>(conv_0_conv_out, output_path+"conv_0_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_0_OCH_PF * CONV_0_OUT_BIT> > conv_0_layer_out("conv_0_layer_out");
#pragma HLS STREAM variable = conv_0_layer_out depth = 320
max_pool2x2<CONV_0_IN_H, CONV_0_IN_W, CONV_0_OUT_CH, CONV_0_OUT_BIT,
            CONV_0_OCH_PF>(conv_0_conv_out, conv_0_layer_out, reps);
#ifdef DEBUG
cout << "conv_0_pool_out size " << conv_0_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_0_IN_H / 2, CONV_0_IN_W / 2,
                           CONV_0_OUT_CH, CONV_0_OCH_PF, CONV_0_OUT_BIT>(conv_0_layer_out, output_path+"conv_0_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 1********************************************************************************/

//--------------------Conv 1: Parameters--------------------
stream<ap_uint<CONV_1_IN_PE * CONV_1_IN_BIT> > &conv_1_in = conv_0_layer_out;
const unsigned CONV_1_M_BIT = CONV_1_IN_BIT + CONV_1_W_BIT + 8;
const unsigned CONV_1_SIMD_BIT = 5;
const unsigned CONV_1_CASCADE = 4;
const unsigned CONV_1_ROW_LEN = (CONV_1_IN_W + CONV_1_K - 1 - 1) / CONV_1_Np + 1;
const unsigned CONV_1_adW_BIT = 1;
const unsigned CONV_1_OCH_PF = CONV_1_PE;
const unsigned CONV_1_DEC_BW_NUM = CONV_1_IN_H * (CONV_1_OUT_CH / CONV_1_OCH_PF) * CONV_1_ROW_LEN;
const unsigned CONV_1_INC_BW_NUM = CONV_1_IN_H * (CONV_1_OUT_CH / CONV_1_OCH_PF) * CONV_1_IN_W * (CONV_1_OCH_PF / CONV_1_ACTP);
const unsigned CONV_1_W_Sep = 1;
const unsigned CONV_1_A_Sep = 1;
    
//--------------------Conv 1: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_1_Np * CONV_1_SIMD * CONV_1_IN_BIT> > conv_1_padding_out("conv_1_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_1_K, CONV_1_IN_H, CONV_1_IN_W, CONV_1_IN_CH, CONV_1_OUT_CH / CONV_1_OCH_PF,
                             CONV_1_Np, CONV_1_IN_BIT, CONV_1_IN_PE, CONV_1_SIMD, 3,
                             8, 2, 2, 2, 7,
                             2, 2>(conv_1_in, conv_1_padding_out, reps);
    
//--------------------Conv 1: Computing Array--------------------
stream<ap_uint<CONV_1_Np * CONV_1_OCH_PF * CONV_1_M_BIT> > conv_1_array_out("conv_1_array_out");
FP_Array<CONV_1_K, CONV_1_ROW_LEN, CONV_1_IN_H, CONV_1_IN_CH, CONV_1_OUT_CH,
         CONV_1_IN_BIT, CONV_1_W_BIT, CONV_1_SIMD * CONV_1_KPF, CONV_1_PE, CONV_1_Kp,
         CONV_1_Np, CONV_1_CASCADE, CONV_1_GUARD_BIT, CONV_1_M_BIT, 
         CONV_1_SIMD_BIT, CONV_1_adW_BIT, CONV_1_W_Sep, CONV_1_A_Sep,
         2, 2, 2, 5>(conv_1_padding_out, conv_1_w, conv_1_array_out, reps);
    
//--------------------Conv 1: Decrease Bit-width--------------------
stream<ap_uint<CONV_1_ACTP * CONV_1_M_BIT> > conv_1_dec_bw_out("conv_1_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_1_Np * CONV_1_OCH_PF * CONV_1_M_BIT, CONV_1_ACTP * CONV_1_M_BIT,
                                  CONV_1_DEC_BW_NUM>(conv_1_array_out, conv_1_dec_bw_out, reps);
    
//--------------------Conv 1: Activate and Trim--------------------
stream<ap_uint<CONV_1_ACTP * CONV_1_OUT_BIT> > conv_1_act_out("conv_1_act_out");
Activation_Trim<CONV_1_K, CONV_1_IN_W, CONV_1_ROW_LEN, CONV_1_IN_H, CONV_1_OUT_CH,
CONV_1_IN_BIT, CONV_1_OUT_BIT, CONV_1_W_BIT, CONV_1_INC_BIT, CONV_1_BIAS_BIT,
CONV_1_L_SHIFT, CONV_1_OCH_PF, CONV_1_ACTP, CONV_1_Np, CONV_1_M_BIT,
2, 8, 5>(conv_1_dec_bw_out, conv_1_inc, conv_1_bias, conv_1_act_out, reps);
    
//--------------------Conv 1: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_1_OCH_PF * CONV_1_OUT_BIT> > conv_1_conv_out("conv_1_conv_out");
#pragma HLS STREAM variable = conv_1_conv_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_1_ACTP * CONV_1_OUT_BIT, 2 * CONV_1_OCH_PF * CONV_1_OUT_BIT,
CONV_1_INC_BW_NUM>(conv_1_act_out, conv_1_conv_out, reps);

#ifdef DEBUG
cout << "conv_1_conv_out size " << conv_1_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_1_IN_H, CONV_1_IN_W, CONV_1_OUT_CH, CONV_1_OCH_PF,
                                    CONV_2_IN_BIT>(conv_1_conv_out, output_path+"conv_1_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_1_OCH_PF * CONV_1_OUT_BIT> > conv_1_layer_out("conv_1_layer_out");
#pragma HLS STREAM variable = conv_1_layer_out depth = 640
max_pool2x2<CONV_1_IN_H, CONV_1_IN_W, CONV_1_OUT_CH, CONV_1_OUT_BIT,
            CONV_1_OCH_PF>(conv_1_conv_out, conv_1_layer_out, reps);
#ifdef DEBUG
cout << "conv_1_pool_out size " << conv_1_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_1_IN_H / 2, CONV_1_IN_W / 2,
                           CONV_1_OUT_CH, CONV_1_OCH_PF, CONV_1_OUT_BIT>(conv_1_layer_out, output_path+"conv_1_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 2********************************************************************************/

//--------------------Conv 2: Parameters--------------------
stream<ap_uint<CONV_2_IN_PE * CONV_2_IN_BIT> > &conv_2_in = conv_1_layer_out;
const unsigned CONV_2_M_BIT = CONV_2_IN_BIT + CONV_2_W_BIT + 9;
const unsigned CONV_2_SIMD_BIT = 6;
const unsigned CONV_2_ROW_LEN = (CONV_2_IN_W + CONV_2_K - 1 - 1) / CONV_2_Np + 1;
const unsigned CONV_2_OCH_PF = CONV_2_PE;
const unsigned CONV_2_DEC_BW_NUM = CONV_2_IN_H * (CONV_2_OUT_CH / CONV_2_OCH_PF) * CONV_2_ROW_LEN;
const unsigned CONV_2_INC_BW_NUM = CONV_2_IN_H * (CONV_2_OUT_CH / CONV_2_OCH_PF) * CONV_2_IN_W * (CONV_2_OCH_PF / CONV_2_ACTP);
    
//--------------------Conv 2: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_2_Np * CONV_2_SIMD * CONV_2_IN_BIT> > conv_2_padding_out("conv_2_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_2_K, CONV_2_IN_H, CONV_2_IN_W, CONV_2_IN_CH, CONV_2_OUT_CH / CONV_2_OCH_PF,
                             CONV_2_Np, CONV_2_IN_BIT, CONV_2_IN_PE, CONV_2_SIMD, 3,
                             7, 2, 4, 2, 6,
                             2, 2>(conv_2_in, conv_2_padding_out, reps);
    
//--------------------Conv 2: Computing Array--------------------
stream<ap_uint<CONV_2_Np * CONV_2_OCH_PF * CONV_2_M_BIT> > conv_2_array_out("conv_2_array_out");
FP_Array_lut<CONV_2_K, CONV_2_ROW_LEN, CONV_2_IN_H, CONV_2_IN_CH, CONV_2_OUT_CH,
         	 CONV_2_IN_BIT, CONV_2_W_BIT, CONV_2_SIMD * CONV_2_KPF, CONV_2_PE, CONV_2_Kp,
         	 CONV_2_Np, CONV_2_M_BIT, CONV_2_SIMD_BIT, 2, 2, 2, 7>(conv_2_padding_out, conv_2_w, conv_2_array_out, reps);
    
//--------------------Conv 2: Decrease Bit-width--------------------
stream<ap_uint<CONV_2_ACTP * CONV_2_M_BIT> > conv_2_dec_bw_out("conv_2_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_2_Np * CONV_2_OCH_PF * CONV_2_M_BIT, CONV_2_ACTP * CONV_2_M_BIT,
                                  CONV_2_DEC_BW_NUM>(conv_2_array_out, conv_2_dec_bw_out, reps);
    
//--------------------Conv 2: Activate and Trim--------------------
stream<ap_uint<CONV_2_ACTP * CONV_2_OUT_BIT> > conv_2_act_out("conv_2_act_out");
Activation_Trim<CONV_2_K, CONV_2_IN_W, CONV_2_ROW_LEN, CONV_2_IN_H, CONV_2_OUT_CH,
CONV_2_IN_BIT, CONV_2_OUT_BIT, CONV_2_W_BIT, CONV_2_INC_BIT, CONV_2_BIAS_BIT,
CONV_2_L_SHIFT, CONV_2_OCH_PF, CONV_2_ACTP, CONV_2_Np, CONV_2_M_BIT,
2, 7, 7>(conv_2_dec_bw_out, conv_2_inc, conv_2_bias, conv_2_act_out, reps);
    
//--------------------Conv 2: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_2_OCH_PF * CONV_2_OUT_BIT> > conv_2_conv_out("conv_2_conv_out");
#pragma HLS STREAM variable = conv_2_conv_out depth = 2560
StreamingDataWidthConverter_Batch<CONV_2_ACTP * CONV_2_OUT_BIT, 2 * CONV_2_OCH_PF * CONV_2_OUT_BIT,
CONV_2_INC_BW_NUM>(conv_2_act_out, conv_2_conv_out, reps);

#ifdef DEBUG
cout << "conv_2_conv_out size " << conv_2_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_2_IN_H, CONV_2_IN_W, CONV_2_OUT_CH, CONV_2_OCH_PF,
                                    CONV_3_IN_BIT>(conv_2_conv_out, output_path+"conv_2_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_2_OCH_PF * CONV_2_OUT_BIT> > conv_2_layer_out("conv_2_layer_out");
#pragma HLS STREAM variable = conv_2_layer_out depth = 1280
max_pool2x2<CONV_2_IN_H, CONV_2_IN_W, CONV_2_OUT_CH, CONV_2_OUT_BIT,
            CONV_2_OCH_PF>(conv_2_conv_out, conv_2_layer_out, reps);
#ifdef DEBUG
cout << "conv_2_pool_out size " << conv_2_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_2_IN_H / 2, CONV_2_IN_W / 2,
                           CONV_2_OUT_CH, CONV_2_OCH_PF, CONV_2_OUT_BIT>(conv_2_layer_out, output_path+"conv_2_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 3********************************************************************************/

//--------------------Conv 3: Parameters--------------------
stream<ap_uint<CONV_3_IN_PE * CONV_3_IN_BIT> > &conv_3_in = conv_2_layer_out;
const unsigned CONV_3_M_BIT = CONV_3_IN_BIT + CONV_3_W_BIT + 10;
const unsigned CONV_3_SIMD_BIT = 5;
const unsigned CONV_3_ROW_LEN = (CONV_3_IN_W + CONV_3_K - 1 - 1) / CONV_3_Np + 1;
const unsigned CONV_3_OCH_PF = CONV_3_PE;
const unsigned CONV_3_DEC_BW_NUM = CONV_3_IN_H * (CONV_3_OUT_CH / CONV_3_OCH_PF) * CONV_3_ROW_LEN;
const unsigned CONV_3_INC_BW_NUM = CONV_3_IN_H * (CONV_3_OUT_CH / CONV_3_OCH_PF) * CONV_3_IN_W * (CONV_3_OCH_PF / CONV_3_ACTP);
    
//--------------------Conv 3: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_3_Np * CONV_3_SIMD * CONV_3_IN_BIT> > conv_3_padding_out("conv_3_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_3_K, CONV_3_IN_H, CONV_3_IN_W, CONV_3_IN_CH, CONV_3_OUT_CH / CONV_3_OCH_PF,
                             CONV_3_Np, CONV_3_IN_BIT, CONV_3_IN_PE, CONV_3_SIMD, 3,
                             6, 2, 4, 3, 7,
                             2, 3>(conv_3_in, conv_3_padding_out, reps);
    
//--------------------Conv 3: Computing Array--------------------
stream<ap_uint<CONV_3_Np * CONV_3_OCH_PF * CONV_3_M_BIT> > conv_3_array_out("conv_3_array_out");
FP_Array_lut<CONV_3_K, CONV_3_ROW_LEN, CONV_3_IN_H, CONV_3_IN_CH, CONV_3_OUT_CH,
         	 CONV_3_IN_BIT, CONV_3_W_BIT, CONV_3_SIMD * CONV_3_KPF, CONV_3_PE, CONV_3_Kp,
         	 CONV_3_Np, CONV_3_M_BIT, CONV_3_SIMD_BIT, 2, 4, 2, 9>(conv_3_padding_out, conv_3_w, conv_3_array_out, reps);
    
//--------------------Conv 3: Decrease Bit-width--------------------
stream<ap_uint<CONV_3_ACTP * CONV_3_M_BIT> > conv_3_dec_bw_out("conv_3_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_3_Np * CONV_3_OCH_PF * CONV_3_M_BIT, CONV_3_ACTP * CONV_3_M_BIT,
                                  CONV_3_DEC_BW_NUM>(conv_3_array_out, conv_3_dec_bw_out, reps);
    
//--------------------Conv 3: Activate and Trim--------------------
stream<ap_uint<CONV_3_ACTP * CONV_3_OUT_BIT> > conv_3_act_out("conv_3_act_out");
Activation_Trim<CONV_3_K, CONV_3_IN_W, CONV_3_ROW_LEN, CONV_3_IN_H, CONV_3_OUT_CH,
CONV_3_IN_BIT, CONV_3_OUT_BIT, CONV_3_W_BIT, CONV_3_INC_BIT, CONV_3_BIAS_BIT,
CONV_3_L_SHIFT, CONV_3_OCH_PF, CONV_3_ACTP, CONV_3_Np, CONV_3_M_BIT,
2, 6, 7>(conv_3_dec_bw_out, conv_3_inc, conv_3_bias, conv_3_act_out, reps);
    
//--------------------Conv 3: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_3_OCH_PF * CONV_3_OUT_BIT> > conv_3_conv_out("conv_3_conv_out");
#pragma HLS STREAM variable = conv_3_conv_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_3_ACTP * CONV_3_OUT_BIT, 2 * CONV_3_OCH_PF * CONV_3_OUT_BIT,
CONV_3_INC_BW_NUM>(conv_3_act_out, conv_3_conv_out, reps);

#ifdef DEBUG
cout << "conv_3_conv_out size " << conv_3_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_3_IN_H, CONV_3_IN_W, CONV_3_OUT_CH, CONV_3_OCH_PF,
                                    CONV_4_IN_BIT>(conv_3_conv_out, output_path+"conv_3_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_3_OCH_PF * CONV_3_OUT_BIT> > conv_3_layer_out("conv_3_layer_out");
#pragma HLS STREAM variable = conv_3_layer_out depth = 640
max_pool2x2<CONV_3_IN_H, CONV_3_IN_W, CONV_3_OUT_CH, CONV_3_OUT_BIT,
            CONV_3_OCH_PF>(conv_3_conv_out, conv_3_layer_out, reps);
#ifdef DEBUG
cout << "conv_3_pool_out size " << conv_3_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_3_IN_H / 2, CONV_3_IN_W / 2,
                           CONV_3_OUT_CH, CONV_3_OCH_PF, CONV_3_OUT_BIT>(conv_3_layer_out, output_path+"conv_3_pool_out.txt", reps);
#endif


/********************************************************************************Convolution 4********************************************************************************/

//--------------------Conv 4: Parameters--------------------
stream<ap_uint<CONV_4_IN_PE * CONV_4_IN_BIT> > &conv_4_in = conv_3_layer_out;
const unsigned CONV_4_M_BIT = CONV_4_IN_BIT + CONV_4_W_BIT + 10;
const unsigned CONV_4_SIMD_BIT = 5;
const unsigned CONV_4_ROW_LEN = (CONV_4_IN_W + CONV_4_K - 1 - 1) / CONV_4_Np + 1;
const unsigned CONV_4_OCH_PF = CONV_4_PE;
const unsigned CONV_4_DEC_BW_NUM = CONV_4_IN_H * (CONV_4_OUT_CH / CONV_4_OCH_PF) * CONV_4_ROW_LEN;
const unsigned CONV_4_INC_BW_NUM = CONV_4_IN_H * (CONV_4_OUT_CH / CONV_4_OCH_PF) * CONV_4_IN_W * (CONV_4_OCH_PF / CONV_4_ACTP);
    
//--------------------Conv 4: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_4_Np * CONV_4_K * CONV_4_SIMD * CONV_4_IN_BIT> > conv_4_padding_out("conv_4_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_4_K, CONV_4_IN_H, CONV_4_IN_W, CONV_4_IN_CH, CONV_4_OUT_CH / CONV_4_OCH_PF,
                             CONV_4_Np, CONV_4_IN_BIT, CONV_4_IN_PE, CONV_4_SIMD, 3,
                             5, 2, 2, 5, 8, 5>(conv_4_in, conv_4_padding_out, reps);
    
//--------------------Conv 4: Computing Array--------------------
stream<ap_uint<CONV_4_Np * CONV_4_OCH_PF * CONV_4_M_BIT> > conv_4_array_out("conv_4_array_out");
FP_Array_lut<CONV_4_K, CONV_4_ROW_LEN, CONV_4_IN_H, CONV_4_IN_CH, CONV_4_OUT_CH,
         	 CONV_4_IN_BIT, CONV_4_W_BIT, CONV_4_SIMD * CONV_4_KPF, CONV_4_PE, CONV_4_Kp,
         	 CONV_4_Np, CONV_4_M_BIT, CONV_4_SIMD_BIT, 2, 5, 2, 11>(conv_4_padding_out, conv_4_w, conv_4_array_out, reps);
    
//--------------------Conv 4: Decrease Bit-width--------------------
stream<ap_uint<CONV_4_ACTP * CONV_4_M_BIT> > conv_4_dec_bw_out("conv_4_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_4_Np * CONV_4_OCH_PF * CONV_4_M_BIT, CONV_4_ACTP * CONV_4_M_BIT,
                                  CONV_4_DEC_BW_NUM>(conv_4_array_out, conv_4_dec_bw_out, reps);
    
//--------------------Conv 4: Activate and Trim--------------------
stream<ap_uint<CONV_4_ACTP * CONV_4_OUT_BIT> > conv_4_act_out("conv_4_act_out");
Activation_Trim<CONV_4_K, CONV_4_IN_W, CONV_4_ROW_LEN, CONV_4_IN_H, CONV_4_OUT_CH,
CONV_4_IN_BIT, CONV_4_OUT_BIT, CONV_4_W_BIT, CONV_4_INC_BIT, CONV_4_BIAS_BIT,
CONV_4_L_SHIFT, CONV_4_OCH_PF, CONV_4_ACTP, CONV_4_Np, CONV_4_M_BIT,
2, 5, 7>(conv_4_dec_bw_out, conv_4_inc, conv_4_bias, conv_4_act_out, reps);
    
//--------------------Conv 4: Increase Bit-width--------------------
stream<ap_uint<CONV_4_OCH_PF * CONV_4_OUT_BIT> > conv_4_layer_out("conv_4_layer_out");
#pragma HLS STREAM variable = conv_4_layer_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_4_ACTP * CONV_4_OUT_BIT, CONV_4_OCH_PF * CONV_4_OUT_BIT, CONV_4_INC_BW_NUM>(conv_4_act_out, conv_4_layer_out, reps);

#ifdef DEBUG
cout << "conv_4_layer_out size " << conv_4_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_4_IN_H, CONV_4_IN_W, CONV_4_OUT_CH, CONV_4_OCH_PF,
                                 CONV_5_IN_BIT>(conv_4_layer_out, output_path+"conv_4_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 5********************************************************************************/

//--------------------Conv 5: Parameters--------------------
stream<ap_uint<CONV_5_IN_PE * CONV_5_IN_BIT> > &conv_5_in = conv_4_layer_out;
const unsigned CONV_5_M_BIT = CONV_5_IN_BIT + CONV_5_W_BIT + 10;
const unsigned CONV_5_SIMD_BIT = 5;
const unsigned CONV_5_CASCADE = 4;
const unsigned CONV_5_ROW_LEN = (CONV_5_IN_W + CONV_5_K - 1 - 1) / CONV_5_Np + 1;
const unsigned CONV_5_adW_BIT = 1;
const unsigned CONV_5_OCH_PF = CONV_5_PE;
const unsigned CONV_5_DEC_BW_NUM = CONV_5_IN_H * (CONV_5_OUT_CH / CONV_5_OCH_PF) * CONV_5_ROW_LEN;
const unsigned CONV_5_INC_BW_NUM = CONV_5_IN_H * (CONV_5_OUT_CH / CONV_5_OCH_PF) * CONV_5_IN_W * (CONV_5_OCH_PF / CONV_5_ACTP);
const unsigned CONV_5_W_Sep = 1;
const unsigned CONV_5_A_Sep = 1;
    
//--------------------Conv 5: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_5_Np * CONV_5_K * CONV_5_SIMD * CONV_5_IN_BIT> > conv_5_padding_out("conv_5_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_5_K, CONV_5_IN_H, CONV_5_IN_W, CONV_5_IN_CH, CONV_5_OUT_CH / CONV_5_OCH_PF,
                             CONV_5_Np, CONV_5_IN_BIT, CONV_5_IN_PE, CONV_5_SIMD, 3,
                             5, 2, 3, 5, 8, 5>(conv_5_in, conv_5_padding_out, reps);
    
//--------------------Conv 5: Computing Array--------------------
stream<ap_uint<CONV_5_Np * CONV_5_OCH_PF * CONV_5_M_BIT> > conv_5_array_out("conv_5_array_out");
FP_Array<CONV_5_K, CONV_5_ROW_LEN, CONV_5_IN_H, CONV_5_IN_CH, CONV_5_OUT_CH,
         CONV_5_IN_BIT, CONV_5_W_BIT, CONV_5_SIMD * CONV_5_KPF, CONV_5_PE, CONV_5_Kp,
         CONV_5_Np, CONV_5_CASCADE, CONV_5_GUARD_BIT, CONV_5_M_BIT, 
         CONV_5_SIMD_BIT, CONV_5_adW_BIT, CONV_5_W_Sep, CONV_5_A_Sep,
         2, 5, 2, 11>(conv_5_padding_out, conv_5_w, conv_5_array_out, reps);
    
//--------------------Conv 5: Decrease Bit-width--------------------
stream<ap_uint<CONV_5_ACTP * CONV_5_M_BIT> > conv_5_dec_bw_out("conv_5_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_5_Np * CONV_5_OCH_PF * CONV_5_M_BIT, CONV_5_ACTP * CONV_5_M_BIT,
                                  CONV_5_DEC_BW_NUM>(conv_5_array_out, conv_5_dec_bw_out, reps);
    
//--------------------Conv 5: Activate and Trim--------------------
stream<ap_uint<CONV_5_ACTP * CONV_5_OUT_BIT> > conv_5_act_out("conv_5_act_out");
Activation_Trim<CONV_5_K, CONV_5_IN_W, CONV_5_ROW_LEN, CONV_5_IN_H, CONV_5_OUT_CH,
CONV_5_IN_BIT, CONV_5_OUT_BIT, CONV_5_W_BIT, CONV_5_INC_BIT, CONV_5_BIAS_BIT,
CONV_5_L_SHIFT, CONV_5_OCH_PF, CONV_5_ACTP, CONV_5_Np, CONV_5_M_BIT,
2, 5, 7>(conv_5_dec_bw_out, conv_5_inc, conv_5_bias, conv_5_act_out, reps);
    
//--------------------Conv 5: Increase Bit-width--------------------
stream<ap_uint<CONV_5_OCH_PF * CONV_5_OUT_BIT> > conv_5_layer_out("conv_5_layer_out");
#pragma HLS STREAM variable = conv_5_layer_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_5_ACTP * CONV_5_OUT_BIT, CONV_5_OCH_PF * CONV_5_OUT_BIT, CONV_5_INC_BW_NUM>(conv_5_act_out, conv_5_layer_out, reps);

#ifdef DEBUG
cout << "conv_5_layer_out size " << conv_5_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_5_IN_H, CONV_5_IN_W, CONV_5_OUT_CH, CONV_5_OCH_PF,
                                 CONV_6_IN_BIT>(conv_5_layer_out, output_path+"conv_5_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 6********************************************************************************/

//--------------------Conv 6: Parameters--------------------
stream<ap_uint<CONV_6_IN_PE * CONV_6_IN_BIT> > &conv_6_in = conv_5_layer_out;
const unsigned CONV_6_M_BIT = CONV_6_IN_BIT + CONV_6_W_BIT + 10;
const unsigned CONV_6_SIMD_BIT = 5;
const unsigned CONV_6_ROW_LEN = (CONV_6_IN_W + CONV_6_K - 1 - 1) / CONV_6_Np + 1;
const unsigned CONV_6_OCH_PF = CONV_6_PE;
const unsigned CONV_6_DEC_BW_NUM = CONV_6_IN_H * (CONV_6_OUT_CH / CONV_6_OCH_PF) * CONV_6_ROW_LEN;
const unsigned CONV_6_INC_BW_NUM = CONV_6_IN_H * (CONV_6_OUT_CH / CONV_6_OCH_PF) * CONV_6_IN_W * (CONV_6_OCH_PF / CONV_6_ACTP);
    
//--------------------Conv 6: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_6_Np * CONV_6_K * CONV_6_SIMD * CONV_6_IN_BIT> > conv_6_padding_out("conv_6_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_6_K, CONV_6_IN_H, CONV_6_IN_W, CONV_6_IN_CH, CONV_6_OUT_CH / CONV_6_OCH_PF,
                             CONV_6_Np, CONV_6_IN_BIT, CONV_6_IN_PE, CONV_6_SIMD, 3,
                             5, 2, 3, 5, 8, 5>(conv_6_in, conv_6_padding_out, reps);
    
//--------------------Conv 6: Computing Array--------------------
stream<ap_uint<CONV_6_Np * CONV_6_OCH_PF * CONV_6_M_BIT> > conv_6_array_out("conv_6_array_out");
FP_Array_lut<CONV_6_K, CONV_6_ROW_LEN, CONV_6_IN_H, CONV_6_IN_CH, CONV_6_OUT_CH,
         	 CONV_6_IN_BIT, CONV_6_W_BIT, CONV_6_SIMD * CONV_6_KPF, CONV_6_PE, CONV_6_Kp,
         	 CONV_6_Np, CONV_6_M_BIT, CONV_6_SIMD_BIT, 2, 5, 2, 11>(conv_6_padding_out, conv_6_w, conv_6_array_out, reps);
    
//--------------------Conv 6: Decrease Bit-width--------------------
stream<ap_uint<CONV_6_ACTP * CONV_6_M_BIT> > conv_6_dec_bw_out("conv_6_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_6_Np * CONV_6_OCH_PF * CONV_6_M_BIT, CONV_6_ACTP * CONV_6_M_BIT,
                                  CONV_6_DEC_BW_NUM>(conv_6_array_out, conv_6_dec_bw_out, reps);
    
//--------------------Conv 6: Activate and Trim--------------------
stream<ap_uint<CONV_6_ACTP * CONV_6_OUT_BIT> > conv_6_act_out("conv_6_act_out");
Activation_Trim<CONV_6_K, CONV_6_IN_W, CONV_6_ROW_LEN, CONV_6_IN_H, CONV_6_OUT_CH,
CONV_6_IN_BIT, CONV_6_OUT_BIT, CONV_6_W_BIT, CONV_6_INC_BIT, CONV_6_BIAS_BIT,
CONV_6_L_SHIFT, CONV_6_OCH_PF, CONV_6_ACTP, CONV_6_Np, CONV_6_M_BIT,
2, 5, 7>(conv_6_dec_bw_out, conv_6_inc, conv_6_bias, conv_6_act_out, reps);
    
//--------------------Conv 6: Increase Bit-width--------------------
stream<ap_uint<CONV_6_OCH_PF * CONV_6_OUT_BIT> > conv_6_layer_out("conv_6_layer_out");
#pragma HLS STREAM variable = conv_6_layer_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_6_ACTP * CONV_6_OUT_BIT, CONV_6_OCH_PF * CONV_6_OUT_BIT, CONV_6_INC_BW_NUM>(conv_6_act_out, conv_6_layer_out, reps);

#ifdef DEBUG
cout << "conv_6_layer_out size " << conv_6_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_6_IN_H, CONV_6_IN_W, CONV_6_OUT_CH, CONV_6_OCH_PF,
                                 CONV_7_IN_BIT>(conv_6_layer_out, output_path+"conv_6_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 7********************************************************************************/

//--------------------Conv 7: Parameters--------------------
stream<ap_uint<CONV_7_IN_PE * CONV_7_IN_BIT> > &conv_7_in = conv_6_layer_out;
const unsigned CONV_7_M_BIT = CONV_7_IN_BIT + CONV_7_W_BIT + 10;
const unsigned CONV_7_SIMD_BIT = 5;
const unsigned CONV_7_CASCADE = 4;
const unsigned CONV_7_ROW_LEN = (CONV_7_IN_W + CONV_7_K - 1 - 1) / CONV_7_Np + 1;
const unsigned CONV_7_adW_BIT = 1;
const unsigned CONV_7_OCH_PF = CONV_7_PE;
const unsigned CONV_7_DEC_BW_NUM = CONV_7_IN_H * (CONV_7_OUT_CH / CONV_7_OCH_PF) * CONV_7_ROW_LEN;
const unsigned CONV_7_INC_BW_NUM = CONV_7_IN_H * (CONV_7_OUT_CH / CONV_7_OCH_PF) * CONV_7_IN_W * (CONV_7_OCH_PF / CONV_7_ACTP);
const unsigned CONV_7_W_Sep = 1;
const unsigned CONV_7_A_Sep = 1;
    
//--------------------Conv 7: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_7_Np * CONV_7_K * CONV_7_SIMD * CONV_7_IN_BIT> > conv_7_padding_out("conv_7_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_7_K, CONV_7_IN_H, CONV_7_IN_W, CONV_7_IN_CH, CONV_7_OUT_CH / CONV_7_OCH_PF,
                             CONV_7_Np, CONV_7_IN_BIT, CONV_7_IN_PE, CONV_7_SIMD, 3,
                             5, 2, 3, 5, 8, 5>(conv_7_in, conv_7_padding_out, reps);
    
//--------------------Conv 7: Computing Array--------------------
stream<ap_uint<CONV_7_Np * CONV_7_OCH_PF * CONV_7_M_BIT> > conv_7_array_out("conv_7_array_out");
FP_Array<CONV_7_K, CONV_7_ROW_LEN, CONV_7_IN_H, CONV_7_IN_CH, CONV_7_OUT_CH,
         CONV_7_IN_BIT, CONV_7_W_BIT, CONV_7_SIMD * CONV_7_KPF, CONV_7_PE, CONV_7_Kp,
         CONV_7_Np, CONV_7_CASCADE, CONV_7_GUARD_BIT, CONV_7_M_BIT, 
         CONV_7_SIMD_BIT, CONV_7_adW_BIT, CONV_7_W_Sep, CONV_7_A_Sep,
         2, 5, 2, 11>(conv_7_padding_out, conv_7_w, conv_7_array_out, reps);
    
//--------------------Conv 7: Decrease Bit-width--------------------
stream<ap_uint<CONV_7_ACTP * CONV_7_M_BIT> > conv_7_dec_bw_out("conv_7_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_7_Np * CONV_7_OCH_PF * CONV_7_M_BIT, CONV_7_ACTP * CONV_7_M_BIT,
                                  CONV_7_DEC_BW_NUM>(conv_7_array_out, conv_7_dec_bw_out, reps);
    
//--------------------Conv 7: Activate and Trim--------------------
stream<ap_uint<CONV_7_ACTP * CONV_7_OUT_BIT> > conv_7_act_out("conv_7_act_out");
Activation_Trim<CONV_7_K, CONV_7_IN_W, CONV_7_ROW_LEN, CONV_7_IN_H, CONV_7_OUT_CH,
CONV_7_IN_BIT, CONV_7_OUT_BIT, CONV_7_W_BIT, CONV_7_INC_BIT, CONV_7_BIAS_BIT,
CONV_7_L_SHIFT, CONV_7_OCH_PF, CONV_7_ACTP, CONV_7_Np, CONV_7_M_BIT,
2, 5, 7>(conv_7_dec_bw_out, conv_7_inc, conv_7_bias, conv_7_act_out, reps);
    
//--------------------Conv 7: Increase Bit-width--------------------
stream<ap_uint<CONV_7_OCH_PF * CONV_7_OUT_BIT> > conv_7_layer_out("conv_7_layer_out");
#pragma HLS STREAM variable = conv_7_layer_out depth = 1280
StreamingDataWidthConverter_Batch<CONV_7_ACTP * CONV_7_OUT_BIT, CONV_7_OCH_PF * CONV_7_OUT_BIT, CONV_7_INC_BW_NUM>(conv_7_act_out, conv_7_layer_out, reps);

#ifdef DEBUG
cout << "conv_7_layer_out size " << conv_7_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_7_IN_H, CONV_7_IN_W, CONV_7_OUT_CH, CONV_7_OCH_PF,
                                 CONV_8_IN_BIT>(conv_7_layer_out, output_path+"conv_7_conv_out.txt", reps);
#endif


/********************************************************************************Convolution 8********************************************************************************/

//--------------------Conv 8: Parameters--------------------
stream<ap_uint<CONV_8_IN_PE * CONV_8_IN_BIT> > &conv_8_in = conv_7_layer_out;
const unsigned CONV_8_M_BIT = 32;
const unsigned CONV_8_SIMD_BIT = 1;
const unsigned CONV_8_CASCADE = 1;
const unsigned CONV_8_ROW_LEN = (CONV_8_IN_W + CONV_8_K - 1 - 1) / CONV_8_Np + 1;
const unsigned CONV_8_adW_BIT = 1;
const bool CONV_8_PatternFlag = false;
const unsigned CONV_8_OCH_PF = CONV_8_PE * CONV_8_Kp;
const unsigned CONV_8_DEC_BW_NUM = CONV_8_IN_H * (CONV_8_OUT_CH / CONV_8_OCH_PF) * CONV_8_ROW_LEN;
const unsigned CONV_8_INC_BW_NUM = CONV_8_IN_H * (CONV_8_OUT_CH / CONV_8_OCH_PF) * CONV_8_IN_W * (CONV_8_OCH_PF / CONV_8_ACTP);
const unsigned CONV_8_W_Sep = 1;
const unsigned CONV_8_A_Sep = 1;
    
//--------------------Conv 8: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_8_Np * CONV_8_SIMD * CONV_8_IN_BIT> > conv_8_padding_out("conv_8_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_8_K, CONV_8_IN_H, CONV_8_IN_W, CONV_8_IN_CH, CONV_8_OUT_CH / CONV_8_OCH_PF,
                             CONV_8_Np, CONV_8_IN_BIT, CONV_8_IN_PE, CONV_8_SIMD, 2,
                             5, 2, 2, 6, 8,
                             2, 6>(conv_8_in, conv_8_padding_out, reps);
    
//--------------------Conv 8: Computing Array--------------------
stream<ap_uint<CONV_8_Np * CONV_8_OCH_PF * CONV_8_M_BIT> > conv_8_array_out("conv_8_array_out");
KP_Array<CONV_8_K, CONV_8_ROW_LEN, CONV_8_IN_H, CONV_8_IN_CH, CONV_8_OUT_CH,
         CONV_8_IN_BIT, CONV_8_W_BIT, CONV_8_SIMD * CONV_8_KPF, CONV_8_PE, CONV_8_Kp,
         CONV_8_Np, CONV_8_CASCADE, CONV_8_GUARD_BIT, CONV_8_M_BIT, 
         CONV_8_SIMD_BIT, CONV_8_adW_BIT, CONV_8_W_Sep, CONV_8_A_Sep, CONV_8_PatternFlag,
         2, 6, 10>(conv_8_padding_out, conv_8_w, conv_8_array_out, reps);
    
//--------------------Conv 8: Decrease Bit-width--------------------
stream<ap_uint<CONV_8_ACTP * CONV_8_M_BIT> > conv_8_dec_bw_out("conv_8_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_8_Np * CONV_8_OCH_PF * CONV_8_M_BIT, CONV_8_ACTP * CONV_8_M_BIT,
                                  CONV_8_DEC_BW_NUM>(conv_8_array_out, conv_8_dec_bw_out, reps);
    
//--------------------Conv 8: Bias and Trim--------------------
stream<ap_uint<CONV_8_ACTP * CONV_8_OUT_BIT> > conv_8_act_out("conv_8_act_out");
Bias_Trim<CONV_8_K, CONV_8_IN_W, CONV_8_ROW_LEN, CONV_8_IN_H, CONV_8_OUT_CH,
CONV_8_OUT_BIT, CONV_8_BIAS_BIT,CONV_8_OCH_PF, CONV_8_ACTP, CONV_8_Np,
2, 5, 5>(conv_8_dec_bw_out, conv_8_bias, conv_8_act_out, reps);
    
//--------------------Conv 8: Increase Bit-width--------------------
stream<ap_uint<CONV_8_OCH_PF * CONV_8_OUT_BIT> > conv_8_layer_out("conv_8_layer_out");
#pragma HLS STREAM variable = conv_8_layer_out depth = 360
StreamingDataWidthConverter_Batch<CONV_8_ACTP * CONV_8_OUT_BIT, CONV_8_OCH_PF * CONV_8_OUT_BIT, CONV_8_INC_BW_NUM>(conv_8_act_out, conv_8_layer_out, reps);

#ifdef DEBUG
cout << "conv_8_layer_out size " << conv_8_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_8_IN_H, CONV_8_IN_W, CONV_8_OUT_CH, CONV_8_OCH_PF,
                                 CONV_8_OUT_BIT>(conv_8_layer_out, output_path+"conv_8_conv_out.txt", reps);
#endif

//-------------------- Add Last --------------------
AddLast<CONV_8_IN_H * CONV_8_IN_W * CONV_8_OUT_CH / 2>(conv_8_layer_out, out, reps);
}

void ultra_net(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
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
#pragma HLS ARRAY_PARTITION variable = conv_8_bias complete dim = 1
  compute_pipeline(in, out, reps);
}


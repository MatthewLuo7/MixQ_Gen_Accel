from string import Template



Front = '''//#define DEBUG
//#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include "config.h"
#include "conv1x1DSP2.hpp"
#include "function.h"
#include "pool_reord.hpp"
#include "stream_tools.h"
#include "weights.hpp"
#include <ap_int.h>
#include <iostream>
#include <stdint.h>
#include "S2P_buffer.hpp"
#include "Opt_FP.hpp"
#include "Opt_KP.hpp"

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

'''

    
Back = '''
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
'''

def get_front():
    return Front

def get_back():
    return Back




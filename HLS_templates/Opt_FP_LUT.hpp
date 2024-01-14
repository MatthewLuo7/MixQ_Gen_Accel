#ifndef __OPT_FP_LUT_HPP__
#define __OPT_FP_LUT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"

template <unsigned IN_BIT, unsigned SIMD, unsigned Np>
void FP_Extract_ACT(ap_uint<Np * SIMD * IN_BIT> in_data, ap_uint<IN_BIT> ipacks[SIMD][Np]) {
#pragma HLS array_partition variable = ipacks dim = 1
#pragma HLS array_partition variable = ipacks dim = 2

  for(unsigned i = 0; i < SIMD; i++){
    for(unsigned j = 0; j < Np; j++){
      ipacks[i][j] = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
    }
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned Kp>
void FP_Extract_W(ap_uint<Kp * SIMD * W_BIT> in_weights, ap_int<W_BIT> wpacks[SIMD][Kp]) {
#pragma HLS array_partition variable = wpacks dim = 1
#pragma HLS array_partition variable = wpacks dim = 2

  for(unsigned i = 0; i < SIMD; i++) {
    for(unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = in_weights(j*SIMD*W_BIT + i*W_BIT + W_BIT - 1, j*SIMD*W_BIT + i*W_BIT);
      wpacks[i][j] = w_seg;
    }
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned Kp, unsigned Np, unsigned ACC_BIT, unsigned SIMD>
void FP_Comp_SIMD_LUT(ap_int<W_BIT> wpacks[SIMD][Kp], ap_uint<IN_BIT> ipacks[SIMD][Np],
                      ap_int<ACC_BIT> DSP_PartialRes[Kp + Np - 1]) {
#pragma HLS ARRAY_PARTITION variable = wpacks dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks dim = 2
#pragma HLS ARRAY_PARTITION variable = ipacks dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks dim = 2
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete
  
  ap_int<ACC_BIT> rtemp[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = rtemp complete
  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    rtemp[i] = 0;
  }

  for(unsigned i = 0; i < SIMD; i++){
  	for(unsigned k = 0; k < Kp; k++){
  		for(unsigned n = 0; n < Np; n++){
  			ap_int<W_BIT + IN_BIT> mul_temp = wpacks[i][k] * ipacks[i][n];

#pragma HLS RESOURCE variable=mul_temp core=Mul_LUT
        mul_temp = wpacks[i][k] * ipacks[i][n];
        
  			rtemp[k + n] += mul_temp;
  		}
  	}
  }

  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    DSP_PartialRes[i] = rtemp[i];
  }
}


template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE, unsigned Kp,
          unsigned Np, unsigned M_BIT, unsigned SIMD_BIT, unsigned k_counter_bw, unsigned infold_counter_bw,
          unsigned res_offset_bw, unsigned add_offset_bw>
void FP_Array_lut(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                  const ap_uint<K * SIMD * W_BIT> weights[PE][(K * IN_CH / SIMD) * (OUT_CH / PE)],
                  stream<ap_uint<Np * PE * M_BIT> > &out,
                  const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned INFOLD = K * IN_CH / SIMD;
  const unsigned KNUM = (K - 1) / Kp + 1;        //ceil(K / Kp)
  const unsigned ACC_BIT = W_BIT + IN_BIT + SIMD_BIT;

  ap_uint<IN_BIT> ipacks[SIMD][Np];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 2

  ap_int<M_BIT> PartialRes[PE][Kp * KNUM + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;
  ap_uint<K * SIMD * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1

  //counters
  ap_uint<k_counter_bw> k_counter = 0;
  ap_uint<infold_counter_bw> infold_counter = 0;
  ap_uint<res_offset_bw> res_offset = 0;
  ap_uint<add_offset_bw> add_offset = 0;       //peIdx * INFOLD

  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < KNUM * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline II = 1

        //flags for input, result reset, and output
        bool flag_in = (k_counter == 0);
        bool flag_res_reset = (infold_counter == 0) && flag_in;
        bool flag_out = ((infold_counter == (INFOLD - 1)) && (k_counter == (KNUM - 1)));

        //input new activations and load weights
        if(flag_in){
          in_data = in.read();
          FP_Extract_ACT<IN_BIT, SIMD, Np>(in_data, ipacks);
          for(unsigned p = 0; p < PE; p++){
            cur_weights[p] = weights[p][add_offset + infold_counter];
          }
        }

        //shift and reset partial result accumulators
        if(flag_res_reset){
          for(unsigned p = 0; p < PE; p++){
            for(unsigned i = 0; i < (K - 1); i++){
              PartialRes[p][i] = PartialRes[p][i + Np];
            }
            for(unsigned j = (K - 1); j < (K + Np - 1); j++){
              PartialRes[p][j] = 0;
            }
          }
        }

        //computing array, PE * SIMD array
        for(unsigned p = 0; p < PE; p++){
          //extract Kp weights
          ap_uint<Kp * SIMD * W_BIT> in_weights = cur_weights[p](Kp * SIMD * W_BIT - 1, 0);
          cur_weights[p] = cur_weights[p] >> (Kp * SIMD * W_BIT);

          ap_int<ACC_BIT> DSP_PartialRes[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1

          ap_int<W_BIT> wpacks[SIMD][Kp];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2
    	    FP_Extract_W<W_BIT, SIMD, Kp>(in_weights, wpacks);

    	    //SIMD computing array
    	    FP_Comp_SIMD_LUT<W_BIT, IN_BIT, Kp, Np, ACC_BIT, SIMD>(wpacks, ipacks, DSP_PartialRes);

          for(unsigned i = 0; i < (Kp + Np - 1); i++){
            PartialRes[p][res_offset + i] += DSP_PartialRes[i];
          }
        }

        //output results
        if(flag_out){
          ap_int<Np * PE * M_BIT> out_data;
          for(unsigned p = 0; p < PE; p++){
            for(unsigned i = 0; i < Np; i++){
              out_data(i*PE*M_BIT + p*M_BIT + M_BIT - 1, i*PE*M_BIT + p*M_BIT) = PartialRes[p][i];
            }
          }
          out.write(out_data);
        }

        //counters
        k_counter++;
        res_offset += Kp;
        if(k_counter == KNUM){
          k_counter = 0;
          res_offset = 0;
          infold_counter++;
          if(infold_counter == INFOLD){
            infold_counter = 0;
          }
        }

        if(cycle == (KNUM * INFOLD * ROW_LEN - 1)){
          add_offset += INFOLD;
          if(add_offset == OUTPENUM * INFOLD){
            add_offset = 0;
          }
        }
      }
    }
  }
}
#endif
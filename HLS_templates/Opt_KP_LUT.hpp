#ifndef __OPT_KP_LUT_HPP__
#define __OPT_KP_LUT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"

template <unsigned IN_BIT, unsigned Np, unsigned SIMD>
void KP_Extract_ACT(ap_uint<Np * SIMD * IN_BIT> in_data,
                 	ap_uint<IN_BIT> ipacks[SIMD][Np]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 2

  for (unsigned i = 0; i < SIMD; i++){
    for (unsigned j = 0; j < Np; j++){
      ipacks[i][j] = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
    }
  }
}

template <unsigned W_BIT, unsigned Kp, unsigned SIMD>
void KP_Extract_W(ap_uint<SIMD * Kp * W_BIT> weights,
               	  ap_int<W_BIT> wpacks[SIMD][Kp]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  for (unsigned i = 0; i < SIMD; i++){
    for (unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = weights(j*W_BIT + i*W_BIT*Kp + W_BIT - 1, j*W_BIT + i*W_BIT*Kp);
      wpacks[i][j] = w_seg;
    }
  }
}


template <unsigned W_BIT, unsigned IN_BIT, unsigned ACC_BIT, unsigned SIMD, unsigned Np, unsigned Kp>
void KP_Comp_SIMD_LUT(ap_uint<IN_BIT> ipacks[SIMD][Np],
                      ap_int<W_BIT> wpacks[SIMD][Kp],
                      ap_int<ACC_BIT> DSP_PartialRes[Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  ap_int<ACC_BIT> res[Kp][Np];
#pragma HLS ARRAY_PARTITION variable = res complete dim = 1
#pragma HLS ARRAY_PARTITION variable = res complete dim = 1

  for(unsigned k = 0; k < Kp; k++){
  	for(unsigned n = 0; n < Np; n++){
  	  res[k][n] = 0;
  	}
  }

  for(unsigned i = 0; i < SIMD; i++){
  	for(unsigned k = 0; k < Kp; k++){
  	  for(unsigned n = 0; n < Np; n++){
#pragma HLS RESOURCE variable=mul_temp core=Mul_LUT
  	    ap_int<W_BIT + IN_BIT> mul_temp = wpacks[i][k] * ipacks[i][n];
  	    res[k][n] += mul_temp;
  	  }
  	}
  }

  // reorder and load results
  for(unsigned i = 0; i < Kp; i++){
    for(unsigned j = 0; j < Np; j++){
      DSP_PartialRes[i][j] = res[i][j];
    }
  }
}


template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE, unsigned Kp,
          unsigned Np, unsigned M_BIT, unsigned SIMD_BIT, unsigned kc_counter_bw, unsigned kich_counter_bw,
          unsigned och_offset_bw>
void KP_Array_lut(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                  const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],      // dim2: Kc --> Kr * IN_CH / SIMD --> OUT_CH / (Kp * PE)
                  stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
                  const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned OUTPENUM = OUT_CH / (PE * Kp);
  const unsigned INFOLD = K * IN_CH / SIMD;
  const unsigned ACC_BIT = W_BIT + IN_BIT + SIMD_BIT;

  ap_uint<IN_BIT> ipacks[SIMD][Np];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 2

  ap_int<M_BIT> PartialRes[Kp * PE][K + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;

  //counters
  ap_uint<kc_counter_bw> kc_counter = 0;
  ap_uint<kich_counter_bw> kich_counter = 0;
  ap_uint<och_offset_bw> och_offset = 0;       //peIdx * K*INFOLD
  for (unsigned h = 0; h < IN_H * reps; h++){
    for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for (unsigned cycle = 0; cycle < K * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline II = 1

        //flags for input, result reset, and output
        bool flag_in = (kc_counter == 0);
        bool flag_res_reset = (kich_counter == 0);
        bool flag_out = (kich_counter == (K*INFOLD - 1));

        //input and pack activations
        if(flag_in){
          in_data = in.read();
          KP_Extract_ACT<IN_BIT, Np, SIMD>(in_data, ipacks);
        }

        //shift and reset partial result accumulators
        if(flag_res_reset){
          for(unsigned p = 0; p < Kp * PE; p++){
            for(unsigned i = 0; i < (K - 1); i++){
              PartialRes[p][i] = PartialRes[p][i + Np];
            }
            for(unsigned j = (K - 1); j < (K + Np - 1); j++){
              PartialRes[p][j] = 0;
            }
          }
        }

        for(unsigned p = 0; p < PE; p++){
          ap_int<ACC_BIT> DSP_PartialRes[Kp][Np];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

          ap_int<W_BIT> wpacks[SIMD][Kp];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2
    	  KP_Extract_W<W_BIT, Kp, SIMD>(weights[p][och_offset + kich_counter], wpacks);

    	  //SIMD computing array
    	  KP_Comp_SIMD_LUT<W_BIT, IN_BIT, ACC_BIT, SIMD, Np, Kp>(ipacks, wpacks, DSP_PartialRes);

          for(unsigned i = 0; i < Kp; i++){
            for(unsigned j = 0; j < Np; j++){
              PartialRes[i + p*Kp][kc_counter + j] += DSP_PartialRes[i][j];
            }
          }
        }

        //output results
        if(flag_out){
          ap_int<Np * Kp * PE * M_BIT> out_data;
          for(unsigned p = 0; p < Kp * PE; p++){
            for(unsigned i = 0; i < Np; i++){
              out_data(i*PE*Kp*M_BIT + p*M_BIT + M_BIT - 1, i*PE*Kp*M_BIT + p*M_BIT) = PartialRes[p][i];
            }
          }
          out.write(out_data);
        }

        //counters
        kc_counter++;
        if(kc_counter == K){
          kc_counter = 0;
        }

        kich_counter++;
        if(kich_counter == K*INFOLD){
          kich_counter = 0;
        }

        if(cycle == (K*INFOLD*ROW_LEN - 1)){
          och_offset += K*INFOLD;
          if(och_offset == OUTPENUM*K*INFOLD){
            och_offset = 0;
          }
        }
      }
    }
  }
}


#endif
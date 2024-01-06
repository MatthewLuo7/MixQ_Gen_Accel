#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// using namespace hls;
// #include <iostream>
using namespace std;
#include "stream_tools.h"
#include <assert.h>

template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu_fixed(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                                 ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT + INC_BIT + 1> bn_res = in * inc + bias;
  const ap_uint<OUT_BIT> res_max = ~0;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > res_max) {
      res = res_max;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}



template <unsigned K, unsigned IN_W, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH,
          unsigned IN_BIT, unsigned OUT_BIT, unsigned W_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned L_SHIFT, unsigned PE, unsigned ACTP,
          unsigned Np, unsigned M_BIT, unsigned ACTP_NUM_counter_bw,
          unsigned w_counter_bw, unsigned add_offset_bw>
void Activation_Trim( stream<ap_uint<ACTP * M_BIT> > &in,
                      const ap_int<INC_BIT> inc[ACTP][OUT_CH / ACTP],
                      const ap_int<BIAS_BIT> bias[ACTP][OUT_CH / ACTP],
                      stream<ap_uint<ACTP * OUT_BIT> > &out,
                      const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned CONV_OUT_W = Np * ROW_LEN;
  const unsigned ACTP_NUM = PE / ACTP;

  ap_uint<ACTP_NUM_counter_bw> ACTP_NUM_counter = 0;
  ap_uint<w_counter_bw> w_counter = 0;
  ap_uint<add_offset_bw> add_offset = 0;            //peIdx*ACTP_NUM
  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < (ACTP_NUM * CONV_OUT_W); cycle++){
      #pragma HLS pipeline

        bool flag_out = ((w_counter >= (K - 1)) && (w_counter < (K - 1 + IN_W)));

        ap_uint<ACTP * M_BIT> in_data;
        in_data = in.read();

        if(flag_out){
          ap_uint<ACTP * OUT_BIT> out_data;
          for(unsigned i = 0; i < ACTP; i++){
            out_data((i + 1) * OUT_BIT - 1, i * OUT_BIT) = bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>
            (in_data((i + 1) * M_BIT - 1, i * M_BIT), inc[i][ACTP_NUM_counter + add_offset], bias[i][ACTP_NUM_counter + add_offset]);
          }
          out.write(out_data);
        }

        //counters
        ACTP_NUM_counter++;
        if(ACTP_NUM_counter == ACTP_NUM){
          ACTP_NUM_counter = 0;
          w_counter++;
          if(w_counter == CONV_OUT_W){
            w_counter = 0;
            add_offset += ACTP_NUM;
            if(add_offset == OUTPENUM * ACTP_NUM){
              add_offset = 0;
            }
          }
        }
      }
    }
  } 
}




template <unsigned K, unsigned IN_W, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH,
          unsigned OUT_BIT, unsigned BIAS_BIT, unsigned PE, unsigned ACTP, unsigned Np,
          unsigned ACTP_NUM_counter_bw, unsigned w_counter_bw, unsigned add_offset_bw>
void Bias_Trim(stream<ap_uint<ACTP * OUT_BIT> > &in,
               const ap_int<BIAS_BIT> bias[ACTP][OUT_CH / ACTP],
               stream<ap_uint<ACTP * OUT_BIT> > &out,
               const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned CONV_OUT_W = Np * ROW_LEN;
  const unsigned ACTP_NUM = PE / ACTP;

  ap_uint<ACTP_NUM_counter_bw> ACTP_NUM_counter = 0;
  ap_uint<w_counter_bw> w_counter = 0;
  ap_uint<add_offset_bw> add_offset = 0;            //peIdx*ACTP_NUM
  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < (ACTP_NUM * CONV_OUT_W); cycle++){
      #pragma HLS pipeline

        bool flag_out = ((w_counter >= (K - 1)) && (w_counter < (K - 1 + IN_W)));

        ap_uint<ACTP * OUT_BIT> in_data;
        in_data = in.read();

        if(flag_out){
          ap_uint<ACTP * OUT_BIT> out_data;
          for(unsigned i = 0; i < ACTP; i++){
            ap_int<OUT_BIT> add_temp = in_data((i + 1) * OUT_BIT - 1, i * OUT_BIT);
            add_temp += bias[i][ACTP_NUM_counter + add_offset];
            out_data((i + 1) * OUT_BIT - 1, i * OUT_BIT) = add_temp;
          }
          out.write(out_data);
        }

        //counters
        ACTP_NUM_counter++;
        if(ACTP_NUM_counter == ACTP_NUM){
          ACTP_NUM_counter = 0;
          w_counter++;
          if(w_counter == CONV_OUT_W){
            w_counter = 0;
            add_offset += ACTP_NUM;
            if(add_offset == OUTPENUM * ACTP_NUM){
              add_offset = 0;
            }
          }
        }
      }
    }
  } 
}


template <unsigned K, unsigned IN_W, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH,
          unsigned OUT_BIT, unsigned PE, unsigned ACTP, unsigned Np, unsigned ACTP_NUM_counter_bw,
          unsigned w_counter_bw, unsigned add_offset_bw>
void Trim(stream<ap_uint<ACTP * OUT_BIT> > &in,
          stream<ap_uint<ACTP * OUT_BIT> > &out,
          const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned CONV_OUT_W = Np * ROW_LEN;
  const unsigned ACTP_NUM = PE / ACTP;

  ap_uint<ACTP_NUM_counter_bw> ACTP_NUM_counter = 0;
  ap_uint<w_counter_bw> w_counter = 0;
  ap_uint<add_offset_bw> add_offset = 0;            //peIdx*ACTP_NUM
  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < (ACTP_NUM * CONV_OUT_W); cycle++){
      #pragma HLS pipeline

        bool flag_out = ((w_counter >= (K - 1)) && (w_counter < (K - 1 + IN_W)));

        ap_uint<ACTP * OUT_BIT> in_data;
        in_data = in.read();

        if(flag_out){
          ap_uint<ACTP * OUT_BIT> out_data;
          for(unsigned i = 0; i < ACTP; i++){
            ap_int<OUT_BIT> add_temp = in_data((i + 1) * OUT_BIT - 1, i * OUT_BIT);
            out_data((i + 1) * OUT_BIT - 1, i * OUT_BIT) = add_temp;
          }
          out.write(out_data);
        }

        //counters
        ACTP_NUM_counter++;
        if(ACTP_NUM_counter == ACTP_NUM){
          ACTP_NUM_counter = 0;
          w_counter++;
          if(w_counter == CONV_OUT_W){
            w_counter = 0;
            add_offset += ACTP_NUM;
            if(add_offset == OUTPENUM * ACTP_NUM){
              add_offset = 0;
            }
          }
        }
      }
    }
  } 
}



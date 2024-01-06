#ifndef __OPT_FP_DW_HPP__
#define __OPT_FP_DW_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"
#include "Opt_FP.hpp"

//-------------------------------------------------------- Basic FP --------------------------------------------------------
template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned KPF, unsigned PE, unsigned Kp,
          unsigned Np, unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned KPF_BIT,
          unsigned adW_BIT, unsigned k_counter_bw, unsigned infold_counter_bw, unsigned res_offset_bw, unsigned add_offset_bw>
void FP_Array_bas_DW(stream<ap_uint<Np * KPF * IN_BIT> > &in,
                     const ap_uint<K * KPF * W_BIT> weights[PE][(K / KPF) * (OUT_CH / PE)],
                     stream<ap_uint<Np * PE * M_BIT> > &out,
                     const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = PROD_BIT * (Kp - 1) + W_BIT + adW_BIT;
  const unsigned IPACK_BIT = PROD_BIT * (Np - 1) + IN_BIT;
  const unsigned PENUM = OUT_CH / PE;
  const unsigned INFOLD = K / KPF;
  const unsigned KNUM = (K - 1) / Kp + 1;        //ceil(K / Kp) 
  const bool Overlap_Flag = ((1 << GUARD_BIT) < Kp) && ((1 << GUARD_BIT) < Np);

  ap_uint<IPACK_BIT> ipacks[KPF];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  ap_int<M_BIT> PartialRes[PE][Kp * KNUM + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * KPF * IN_BIT> in_data = 0;
  ap_uint<K * KPF * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1

  //counters
  ap_uint<k_counter_bw> k_counter = 0;
  ap_uint<infold_counter_bw> infold_counter = 0;
  ap_uint<res_offset_bw> res_offset = 0;
  ap_uint<add_offset_bw> add_offset = 0;       //peIdx * INFOLD

  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < PENUM; peIdx++){
      for(unsigned cycle = 0; cycle < KNUM * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline II = 1

        //flags for input, result reset, and output
        bool flag_in = (k_counter == 0);
        bool flag_res_reset = (infold_counter == 0) && flag_in;
        bool flag_out = ((infold_counter == (INFOLD - 1)) && (k_counter == (KNUM - 1)));

        //input new activations and load weights
        if(flag_in){
          in_data = in.read();
          FP_Pack_ACT<IN_BIT, KPF, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks);
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

        //computing array, PE * KPF array
        for(unsigned p = 0; p < PE; p++){
          //extract Kp weights
          ap_uint<Kp * KPF * W_BIT> in_weights = cur_weights[p](Kp * KPF * W_BIT - 1, 0);
          cur_weights[p] = cur_weights[p] >> (Kp * KPF * W_BIT);

          ap_int<W_BIT + IN_BIT + KPF_BIT + 1> DSP_PartialRes[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          FP_Comp_SIMD<Kp, Np, W_BIT, IN_BIT, PROD_BIT, KPF, CASCADE, KPF_BIT, WPACK_BIT, IPACK_BIT, Overlap_Flag>(in_weights, ipacks, DSP_PartialRes);

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
          if(add_offset == PENUM * INFOLD){
            add_offset = 0;
          }
        }
      }
    }
  }
}


//-------------------------------------------------------- Operand Seperation FP --------------------------------------------------------
template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH, unsigned IN_BIT,
          unsigned W_BIT, unsigned KPF, unsigned PE, unsigned Kp, unsigned Np,
          unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned KPF_BIT, unsigned adW_BIT,
          unsigned W_Sep, unsigned A_Sep, unsigned k_counter_bw, unsigned infold_counter_bw, unsigned res_offset_bw,
          unsigned add_offset_bw>
void FP_Array_sep_DW(stream<ap_uint<Np * KPF * IN_BIT> > &in,
                     const ap_uint<K * KPF * W_BIT> weights[PE][(K / KPF) * (OUT_CH  / PE)],
                     stream<ap_uint<Np * PE * M_BIT> > &out,
                     const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const bool Sep_Flag = A_Sep > W_Sep;

  const unsigned IN_BIT_L = IN_BIT / 2;
  const unsigned IN_BIT_H = IN_BIT - IN_BIT_L;
  const unsigned W_BIT_L = W_BIT / 2;
  const unsigned W_BIT_H = W_BIT - W_BIT_L;

  const unsigned M_BIT_Sep = (Sep_Flag) ? (M_BIT - IN_BIT + IN_BIT_H):(M_BIT - W_BIT + W_BIT_H);
  const unsigned PROD_BIT =(Sep_Flag) ? (W_BIT + IN_BIT_H + GUARD_BIT):(W_BIT_H + IN_BIT + GUARD_BIT);
  const unsigned WPACK_BIT = (Sep_Flag) ? (PROD_BIT * (Kp - 1) + W_BIT + adW_BIT):(PROD_BIT * (Kp - 1) + W_BIT_H + adW_BIT);
  const unsigned IPACK_BIT = (Sep_Flag) ? (PROD_BIT * (Np - 1) + IN_BIT_H):(PROD_BIT * (Np - 1) + IN_BIT);
  const unsigned ACC_Left_Shift = (Sep_Flag) ? IN_BIT_L:W_BIT_L;
  const unsigned ACC_BIT = (Sep_Flag) ? (W_BIT + IN_BIT_H + KPF_BIT):(W_BIT_H + IN_BIT + KPF_BIT);
  
  const unsigned PENUM = OUT_CH / PE;
  const unsigned INFOLD = K / KPF;
  const unsigned KNUM = (K - 1) / Kp + 1;        //ceil(K / Kp) 
  const bool Overlap_Flag = ((1 << GUARD_BIT) < Kp) && ((1 << GUARD_BIT) < Np);

  ap_uint<IPACK_BIT> ipacks[A_Sep][KPF];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  ap_int<M_BIT_Sep> PartialRes[2][PE][Kp * KNUM + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 3

  ap_uint<Np * KPF * IN_BIT> in_data = 0;
  ap_uint<K * KPF * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1

  //counters
  ap_uint<k_counter_bw> k_counter = 0;
  ap_uint<infold_counter_bw> infold_counter = 0;
  ap_uint<res_offset_bw> res_offset = 0;
  ap_uint<add_offset_bw> add_offset = 0;       //peIdx * INFOLD

  for(unsigned h = 0; h < IN_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < PENUM; peIdx++){
      for(unsigned cycle = 0; cycle < KNUM * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline II = 1

        //flags for input, result reset, and output
        bool flag_in = (k_counter == 0);
        bool flag_res_reset = (infold_counter == 0) && flag_in;
        bool flag_out = ((infold_counter == (INFOLD - 1)) && (k_counter == (KNUM - 1)));

        //input new activations and load weights
        if(flag_in){
          in_data = in.read();
          if(Sep_Flag){
            FP_Pack_ACT_sep<IN_BIT, IN_BIT_H, IN_BIT_L, KPF, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks);
          }else{
            FP_Pack_ACT<IN_BIT, KPF, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks[0]);
          }
          for(unsigned p = 0; p < PE; p++){
            cur_weights[p] = weights[p][add_offset + infold_counter];
          }
        }

        //shift and reset partial result accumulators
        if(flag_res_reset){
          for(unsigned k = 0; k < 2; k++){
            for(unsigned p = 0; p < PE; p++){
              for(unsigned i = 0; i < (K - 1); i++){
                PartialRes[k][p][i] = PartialRes[k][p][i + Np];
              }
              for(unsigned j = (K - 1); j < (K + Np - 1); j++){
                PartialRes[k][p][j] = 0;
              }
            }
          }  
        }

        //computing array, PE * KPF array
        for(unsigned p = 0; p < PE; p++){
          //extract Kp weights
          ap_uint<Kp * KPF * W_BIT> in_weights = cur_weights[p](Kp * KPF * W_BIT - 1, 0);
          cur_weights[p] = cur_weights[p] >> (Kp * KPF * W_BIT);

          ap_int<ACC_BIT + 1> DSP_PartialRes[2][Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2
          FP_Comp_SIMD_sep<Kp, Np, W_BIT, W_BIT_H, W_BIT_L, IN_BIT, PROD_BIT, KPF, CASCADE, KPF_BIT,
          WPACK_BIT, IPACK_BIT, ACC_BIT, W_Sep, A_Sep, Overlap_Flag, Sep_Flag>(in_weights, ipacks, DSP_PartialRes);

          for(unsigned k = 0; k < 2; k++){
            for(unsigned i = 0; i < (Kp + Np - 1); i++){
              PartialRes[k][p][res_offset + i] += DSP_PartialRes[k][i];
            }
          }  
        }

        //output results
        if(flag_out){
          ap_int<Np * PE * M_BIT> out_data;
          for(unsigned p = 0; p < PE; p++){
            for(unsigned i = 0; i < Np; i++){
              ap_int<M_BIT> out_temp = (PartialRes[1][p][i] * (1 << ACC_Left_Shift)) + PartialRes[0][p][i];
              out_data(i*PE*M_BIT + p*M_BIT + M_BIT - 1, i*PE*M_BIT + p*M_BIT) = out_temp;
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
          if(add_offset == PENUM * INFOLD){
            add_offset = 0;
          }
        }
      }
    }
  }
}


//-------------------------------------------------------- Unified FP Wrapper --------------------------------------------------------
template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned KPF, unsigned PE, unsigned Kp,
          unsigned Np, unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned KPF_BIT,
          unsigned adW_BIT, unsigned W_Sep, unsigned A_Sep, unsigned k_counter_bw, unsigned infold_counter_bw,
          unsigned res_offset_bw, unsigned add_offset_bw>
void FP_Array_DW(stream<ap_uint<Np * KPF * IN_BIT> > &in,
                 const ap_uint<K * KPF * W_BIT> weights[PE][(K / KPF) * (OUT_CH / PE)],
                 stream<ap_uint<Np * PE * M_BIT> > &out,
                 const unsigned reps = 1){
  const unsigned SEL = W_Sep * A_Sep;

  if(SEL == 1){
    FP_Array_bas_DW<K, ROW_LEN, IN_H, OUT_CH, IN_BIT, W_BIT, KPF, PE,
                    Kp, Np, CASCADE, GUARD_BIT, M_BIT, KPF_BIT, adW_BIT,
                    k_counter_bw, infold_counter_bw, res_offset_bw, add_offset_bw>(in, weights, out, reps);
  }else{                                                                                            // SEL == 2
    FP_Array_sep_DW<K, ROW_LEN, IN_H, OUT_CH, IN_BIT, W_BIT, KPF, PE,
                    Kp, Np, CASCADE, GUARD_BIT, M_BIT, KPF_BIT, adW_BIT, W_Sep, A_Sep,
                    k_counter_bw, infold_counter_bw, res_offset_bw, add_offset_bw>(in, weights, out, reps);
  }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

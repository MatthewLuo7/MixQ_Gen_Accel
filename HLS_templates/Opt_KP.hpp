#ifndef __OPT_KP_HPP__
#define __OPT_KP_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"


template <unsigned IN_BIT, unsigned IPACK_BIT, unsigned AITV_BIT, unsigned Np, unsigned SIMD>
void KP_Pack_ACT(ap_uint<Np * SIMD * IN_BIT> in_data,
                 ap_uint<IPACK_BIT> ipacks[SIMD]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  for (unsigned i = 0; i < SIMD; i++){
    ap_uint<IPACK_BIT> temp = 0;
    for (unsigned j = 0; j < Np; j++){
      temp(j*AITV_BIT + IN_BIT - 1, j*AITV_BIT) = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
    }
    ipacks[i] = temp;
  }
}


template <unsigned W_BIT, unsigned WITV_BIT, unsigned WPACK_BIT, unsigned Kp, unsigned SIMD>
void KP_Pack_W(ap_uint<SIMD * Kp * W_BIT> weights,
               ap_int<WPACK_BIT> wpacks[SIMD]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1

  for (unsigned i = 0; i < SIMD; i++){
    ap_int<WPACK_BIT> wpack_temp = 0;
    for (unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = weights(j*W_BIT + i*W_BIT*Kp + W_BIT - 1, j*W_BIT + i*W_BIT*Kp);
      wpack_temp += (w_seg * (1 << (j*WITV_BIT)));
    }
    wpacks[i] = wpack_temp;
  }
}

template <unsigned W_BIT, unsigned WITV_BIT, unsigned WPACK_BIT, unsigned Kp, unsigned SIMD>
void KP_Pack_W_overlap(ap_uint<SIMD * Kp * W_BIT> weights,
                       ap_int<WPACK_BIT> wpacks[SIMD],
                       ap_uint<Kp> wpfix[SIMD]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1

  for (unsigned i = 0; i < SIMD; i++){
    ap_int<WPACK_BIT> wpack_temp = 0;
    for (unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = weights(j*W_BIT + i*W_BIT*Kp + W_BIT - 1, j*W_BIT + i*W_BIT*Kp);
      wpfix[i][j] = w_seg[0];
      wpack_temp += (w_seg * (1 << (j*WITV_BIT)));
    }
    wpacks[i] = wpack_temp;
  }
}



template <unsigned ACC_BIT, unsigned IPACK_BIT, unsigned WPACK_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned Np, unsigned Kp, unsigned CASCADE, bool Pattern_Flag>
void KP_Comp_SIMD_cascade(ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<WPACK_BIT> wpacks[SIMD],
                          ap_int<ACC_BIT + 1> DSP_PartialRes[Kp][Np],
                          bool Is_Signed){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  ap_int<ACC_BIT + 1> res[Kp * Np];
  for (unsigned t = 0; t < (Kp * Np); t++){
    res[t] = 0;
  }

  for (unsigned i = 0; i < SIMD; i += CASCADE){
    ap_int<PROD_BIT * Np * Kp> DSP_Res = 0;
    for (unsigned j = 0; j < CASCADE; j++){           //cascade through DSP's accumulator
      ap_int<PROD_BIT * Np * Kp> DSP;
#pragma HLS RESOURCE variable=DSP core=DSP48
      DSP = ipacks[i+j] * wpacks[i+j];
      DSP_Res += DSP;
    }

    if(Is_Signed){
      ap_int<PROD_BIT> res0_temp = DSP_Res(PROD_BIT - 1, 0);
#pragma HLS RESOURCE variable=res core=AddSubnS
      res[0] += res0_temp;
      for (unsigned k = 1; k < (Kp * Np); k++){
        ap_int<PROD_BIT> res_temp = DSP_Res(k*PROD_BIT + PROD_BIT - 1, k*PROD_BIT);
        ap_int<PROD_BIT> acc_res_temp;
#pragma HLS RESOURCE variable=acc_res_temp core=AddSubnS
        acc_res_temp = res_temp + DSP_Res[k*PROD_BIT - 1];
#pragma HLS RESOURCE variable=res core=AddSubnS
        res[k] += acc_res_temp;
      }
    }else{
      for (unsigned k = 0; k < (Kp * Np); k++){
        ap_uint<PROD_BIT> res_temp = DSP_Res(k*PROD_BIT + PROD_BIT - 1, k*PROD_BIT);
#pragma HLS RESOURCE variable=res core=AddSubnS
        res[k] += res_temp;
      }
    }
  }

  // reorder and load results
  if(Pattern_Flag){                             //weights are packed more densely
    for(unsigned i = 0; i < Kp; i++){
      for(unsigned j = 0; j < Np; j++){
        DSP_PartialRes[i][j] = res[i + j*Kp];
      }
    }
  }else{                                   //activations are packed more densely
    for(unsigned i = 0; i < Kp; i++){
      for(unsigned j = 0; j < Np; j++){
        DSP_PartialRes[i][j] = res[j + i*Np];
      }
    }
  }
}


template <unsigned ACC_BIT, unsigned IPACK_BIT, unsigned WPACK_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned Np, unsigned Kp, bool Pattern_Flag>
void KP_Comp_SIMD_overlap(ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<WPACK_BIT> wpacks[SIMD],
                          ap_uint<Kp> wpfix[SIMD],
                          ap_int<ACC_BIT + 1> DSP_PartialRes[Kp][Np],
                          bool Is_Signed){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  ap_int<ACC_BIT> res[Kp * Np];           //Initialize accumulation variables
  for (unsigned t = 0; t < (Kp * Np); t++){
    res[t] = 0;
  }

  for (unsigned i = 0; i < SIMD; i++){
    ap_uint<1> fixsig[Np*Kp];
#pragma HLS ARRAY_PARTITION variable = fixsig complete dim = 1
    for (unsigned Kp_i = 0; Kp_i < Kp; Kp_i++){             //Calculate LSBs
      for (unsigned Np_i = 0; Np_i < Np; Np_i++){
        ap_uint<1> i_lsb = ipacks[i][PROD_BIT*Np_i];
        ap_uint<1> w_lsb = wpfix[i][Kp_i];
        if (Pattern_Flag){
          fixsig[Kp_i + Np_i*Kp] = w_lsb&i_lsb;
        }
        else{
          fixsig[Np_i + Kp_i*Np] = w_lsb&i_lsb;
        }
      }
    }

    ap_int<PROD_BIT * Np * Kp + 1> DSP_Res;
    #pragma HLS RESOURCE variable=DSP_Res core=DSP48
    DSP_Res = ipacks[i] * wpacks[i];                 //DSP packing multiplication

    if(Is_Signed){
      ap_int<PROD_BIT + 1> rfix[Np*Kp];
#pragma HLS ARRAY_PARTITION variable = rfix complete
  
      rfix[0] = (fixsig[1], (ap_uint<PROD_BIT>) 0);
      rfix[Np*Kp - 1] = ((ap_uint<PROD_BIT>) 0, fixsig[Np*Kp - 1]^DSP_Res[(Np*Kp - 1)*PROD_BIT]);
      for (unsigned j = 1; j < (Np*Kp - 1); j++){
        rfix[j] = (fixsig[j+1], (ap_uint<PROD_BIT - 1>) 0, fixsig[j]^DSP_Res[j*PROD_BIT]);
      }
  
      for (unsigned k = 0; k < Np*Kp; k++){
        ap_int<PROD_BIT + 1> res_temp = DSP_Res((k+1)*PROD_BIT, k*PROD_BIT);
        ap_int<PROD_BIT + 1> acc_res_temp;
#pragma HLS RESOURCE variable=acc_res_temp core=AddSubnS
        acc_res_temp = res_temp + rfix[k];
#pragma HLS RESOURCE variable=res core=AddSubnS
        res[k] += acc_res_temp;
      }
    }else{
      ap_uint<PROD_BIT + 1> rfix[Np*Kp];
#pragma HLS ARRAY_PARTITION variable = rfix complete
  
      rfix[0] = (fixsig[1], (ap_uint<PROD_BIT>) 0);
      rfix[Np*Kp - 1] = (fixsig[Np*Kp - 1] == DSP_Res[(Np*Kp - 1)*PROD_BIT]) ? ~0:0;
      for (unsigned j = 1; j < (Np*Kp - 1); j++){
        ap_uint<PROD_BIT + 1> rfix_temp = fixsig[j]^DSP_Res[j*PROD_BIT];
        rfix_temp += (fixsig[j+1], (ap_uint<PROD_BIT>) 0);
        rfix[j] = rfix_temp;
      }
  
      for (unsigned k = 0; k < Np*Kp; k++){
        ap_uint<PROD_BIT + 1> res_temp = DSP_Res((k+1)*PROD_BIT, k*PROD_BIT);
        ap_uint<PROD_BIT + 1> acc_res_temp;
#pragma HLS RESOURCE variable=acc_res_temp core=AddSubnS
        acc_res_temp = res_temp + rfix[k];
#pragma HLS RESOURCE variable=res core=AddSubnS
        res[k] += acc_res_temp;
      }
    }
  }

  // reorder and load results
  if(Pattern_Flag){                             //weights are packed more densely
    for(unsigned i = 0; i < Kp; i++){
      for(unsigned j = 0; j < Np; j++){
        DSP_PartialRes[i][j] = res[i + j*Kp];
      }
    }
  }else{                                   //activations are packed more densely
    for(unsigned i = 0; i < Kp; i++){
      for(unsigned j = 0; j < Np; j++){
        DSP_PartialRes[i][j] = res[j + i*Np];
      }
    }
  }
}

template <unsigned IN_BIT, unsigned W_BIT, unsigned IPACK_BIT, unsigned WPACK_BIT,
          unsigned PROD_BIT, unsigned SIMD_BIT, unsigned SIMD, unsigned Np,
          unsigned Kp, unsigned CASCADE, unsigned WITV_BIT, bool Overlap_Flag, bool Pattern_Flag>
void KP_Comp_SIMD(ap_uint<SIMD * Kp * W_BIT> weights,               
                  ap_uint<IPACK_BIT> ipacks[SIMD],
                  ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> DSP_PartialRes[Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  const unsigned ACC_BIT = W_BIT + IN_BIT + SIMD_BIT;

  if (Overlap_Flag){
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    ap_uint<Kp> wpfix[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
    KP_Pack_W_overlap<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks, wpfix);

    //SIMD computing array
    KP_Comp_SIMD_overlap<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, Pattern_Flag>(ipacks, wpacks, wpfix, DSP_PartialRes, true);
  }else{
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    KP_Pack_W<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks);

    //SIMD computing array
    KP_Comp_SIMD_cascade<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, CASCADE, Pattern_Flag>(ipacks, wpacks, DSP_PartialRes, true);
  } 
}

template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE, unsigned Kp,
          unsigned Np, unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned SIMD_BIT,
          bool Pattern_Flag, unsigned kc_counter_bw, unsigned kich_counter_bw, unsigned och_offset_bw>
void KP_Array_bas(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                  const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],      // dim2: Kc --> Kr * IN_CH / SIMD --> OUT_CH / (Kp * PE)
                  stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
                  const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned PROD_BIT = IN_BIT + W_BIT + GUARD_BIT;
  const unsigned WITV_BIT = Pattern_Flag ? PROD_BIT : (Np * PROD_BIT);
  const unsigned AITV_BIT = Pattern_Flag ? (Kp * PROD_BIT) : PROD_BIT;
  const unsigned IPACK_BIT = (Np - 1) * AITV_BIT + IN_BIT;
  const unsigned WPACK_BIT = (Kp - 1) * WITV_BIT + W_BIT;
  const unsigned OUTPENUM = OUT_CH / (PE * Kp);
  const unsigned INFOLD = K * IN_CH / SIMD;
  const bool Overlap_Flag = (GUARD_BIT < 0);

  ap_uint<IPACK_BIT> ipacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

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
          KP_Pack_ACT<IN_BIT, IPACK_BIT, AITV_BIT, Np, SIMD>(in_data, ipacks);
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
          ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> DSP_PartialRes[Kp][Np];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2
          KP_Comp_SIMD<IN_BIT, W_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD,
          Np, Kp, CASCADE, WITV_BIT, Overlap_Flag, Pattern_Flag>(weights[p][och_offset + kich_counter], ipacks, DSP_PartialRes);

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

template <unsigned IN_BIT, unsigned IN_BIT_H, unsigned IN_BIT_L, unsigned IPACK_BIT, unsigned AITV_BIT, unsigned Np, unsigned SIMD>
void KP_Pack_ACT_sep(ap_uint<Np * SIMD * IN_BIT> in_data,
                     ap_uint<IPACK_BIT> ipacks[2][SIMD]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 2

  for (unsigned i = 0; i < SIMD; i++){
    ap_uint<IPACK_BIT> temp_h = 0;
    ap_uint<IPACK_BIT> temp_l = 0;
    for (int j = 0; j < Np; j++){
      ap_uint<IN_BIT> data = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
      ap_uint<IN_BIT_H> data_h = data(IN_BIT_L + IN_BIT_H - 1, IN_BIT_L);
      ap_uint<IN_BIT_L> data_l = data(IN_BIT_L - 1, 0);
      temp_h(j*AITV_BIT + IN_BIT_H - 1, j*AITV_BIT) = data_h;
      temp_l(j*AITV_BIT + IN_BIT_L - 1, j*AITV_BIT) = data_l;
    }
    ipacks[0][i] = temp_l;
    ipacks[1][i] = temp_h;
  }
}

template <unsigned W_BIT, unsigned W_BIT_H, unsigned W_BIT_L, unsigned WITV_BIT, unsigned WPACK_BIT, unsigned Kp, unsigned SIMD>
void KP_Pack_W_sep(ap_uint<SIMD * Kp * W_BIT> weights,
                   ap_int<WPACK_BIT> wpacks[2][SIMD]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  for (unsigned i = 0; i < SIMD; i++){
    ap_int<WPACK_BIT> wpack_temp_h = 0;
    ap_int<WPACK_BIT> wpack_temp_l = 0;
    for (unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = weights(j*W_BIT + i*W_BIT*Kp + W_BIT - 1, j*W_BIT + i*W_BIT*Kp);
      ap_int<W_BIT_H> w_seg_h = w_seg(W_BIT_L + W_BIT_H - 1, W_BIT_L);
      ap_uint<W_BIT_L> w_seg_l = w_seg(W_BIT_L - 1, 0);
      wpack_temp_h += (w_seg_h * (1 << (WITV_BIT * j)));
      wpack_temp_l(WITV_BIT * j + W_BIT_L - 1, WITV_BIT * j) = w_seg_l;
    }
    wpacks[0][i] = wpack_temp_l;
    wpacks[1][i] = wpack_temp_h;
  }
}

template <unsigned W_BIT, unsigned W_BIT_H, unsigned W_BIT_L, unsigned WITV_BIT, unsigned WPACK_BIT, unsigned Kp, unsigned SIMD>
void KP_Pack_W_overlap_sep(ap_uint<SIMD * Kp * W_BIT> weights,
                           ap_int<WPACK_BIT> wpacks[2][SIMD],
                           ap_uint<Kp> wpfix[2][SIMD]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 2

  for (unsigned i = 0; i < SIMD; i++){
    ap_int<WPACK_BIT> wpack_temp_h = 0;
    ap_int<WPACK_BIT> wpack_temp_l = 0;
    for (unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = weights(j*W_BIT + i*W_BIT*Kp + W_BIT - 1, j*W_BIT + i*W_BIT*Kp);
      ap_int<W_BIT_H> w_seg_h = w_seg(W_BIT_L + W_BIT_H - 1, W_BIT_L);
      ap_uint<W_BIT_L> w_seg_l = w_seg(W_BIT_L - 1, 0);

      wpack_temp_h += (w_seg_h * (1 << (WITV_BIT * j)));
      wpack_temp_l(WITV_BIT * j + W_BIT_L - 1, WITV_BIT * j) = w_seg_l;

      wpfix[0][i][j] = w_seg_l[0];
      wpfix[1][i][j] = w_seg_h[0];
    }
    wpacks[0][i] = wpack_temp_l;
    wpacks[1][i] = wpack_temp_h;
  }
}



template <unsigned IN_BIT, unsigned W_BIT, unsigned W_BIT_H, unsigned W_BIT_L, unsigned IPACK_BIT,
          unsigned WPACK_BIT, unsigned PROD_BIT, unsigned SIMD_BIT, unsigned SIMD, unsigned Np,
          unsigned Kp, unsigned CASCADE, unsigned WITV_BIT, unsigned ACC_BIT, unsigned W_Sep,
          unsigned A_Sep, bool Pattern_Flag, bool Overlap_Flag, bool Sep_Flag>
void KP_Comp_SIMD_sep(ap_uint<SIMD * Kp * W_BIT> weights,               
                      ap_uint<IPACK_BIT> ipacks[A_Sep][SIMD],
                      ap_int<ACC_BIT + 1> DSP_PartialRes[2][Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 3

  if (Overlap_Flag){
    ap_int<WPACK_BIT> wpacks[W_Sep][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2
    ap_uint<Kp> wpfix[W_Sep][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 2

    if(Sep_Flag){
      KP_Pack_W_overlap<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks[0], wpfix[0]);
      for(unsigned i = 0; i < 2; i++){
        KP_Comp_SIMD_overlap<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, Pattern_Flag>(ipacks[i], wpacks[0], wpfix[0], DSP_PartialRes[i], true);
      }
    }else{
      KP_Pack_W_overlap_sep<W_BIT, W_BIT_H, W_BIT_L, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks, wpfix);
      for(unsigned i = 0; i < 2; i++){
        bool Is_Signed = (i == 0) ? false:true;
        KP_Comp_SIMD_overlap<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, Pattern_Flag>(ipacks[0], wpacks[i], wpfix[i], DSP_PartialRes[i], Is_Signed);
      }
    }
  }else{
    ap_int<WPACK_BIT> wpacks[W_Sep][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

    if(Sep_Flag){
      KP_Pack_W<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks[0]);
      for(unsigned i = 0; i < 2; i++){
        KP_Comp_SIMD_cascade<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, CASCADE, Pattern_Flag>(ipacks[i], wpacks[0], DSP_PartialRes[i], true);
      }
    }else{
      KP_Pack_W_sep<W_BIT, W_BIT_H, W_BIT_L, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks);
      for(unsigned i = 0; i < 2; i++){
        bool Is_Signed = (i == 0) ? false:true;
        KP_Comp_SIMD_cascade<ACC_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD, Np, Kp, CASCADE, Pattern_Flag>(ipacks[0], wpacks[i], DSP_PartialRes[i], Is_Signed);
      }
    }
  } 
}


template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE, unsigned Kp,
          unsigned Np, unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned SIMD_BIT,
          unsigned W_Sep, unsigned A_Sep, bool Pattern_Flag, unsigned kc_counter_bw,
          unsigned kich_counter_bw, unsigned och_offset_bw>
void KP_Array_sep(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                  const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],      // dim2: Kc --> Kr * IN_CH / SIMD --> OUT_CH / (Kp * PE)
                  stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
                  const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const bool Sep_Flag = A_Sep > W_Sep;

  const unsigned IN_BIT_L = IN_BIT / 2;
  const unsigned IN_BIT_H = IN_BIT - IN_BIT_L;
  const unsigned W_BIT_L = W_BIT / 2;
  const unsigned W_BIT_H = W_BIT - W_BIT_L;

  const unsigned M_BIT_Sep = (Sep_Flag) ? (M_BIT - IN_BIT + IN_BIT_H):(M_BIT - W_BIT + W_BIT_H);
  const unsigned PROD_BIT =(Sep_Flag) ? (W_BIT + IN_BIT_H + GUARD_BIT):(W_BIT_H + IN_BIT + GUARD_BIT);
  const unsigned WITV_BIT = Pattern_Flag ? PROD_BIT : (Np * PROD_BIT);
  const unsigned AITV_BIT = Pattern_Flag ? (Kp * PROD_BIT) : PROD_BIT;
  const unsigned IPACK_BIT = (Sep_Flag) ? ((Np - 1) * AITV_BIT + IN_BIT_H):((Np - 1) * AITV_BIT + IN_BIT);
  const unsigned WPACK_BIT = (Sep_Flag) ? ((Kp - 1) * WITV_BIT + W_BIT):((Kp - 1) * WITV_BIT + W_BIT_H);
  const unsigned ACC_Left_Shift = (Sep_Flag) ? IN_BIT_L:W_BIT_L;
  const unsigned ACC_BIT = (Sep_Flag) ? (W_BIT + IN_BIT_H + SIMD_BIT):(W_BIT_H + IN_BIT + SIMD_BIT);

  const unsigned OUTPENUM = OUT_CH / (PE * Kp);
  const unsigned INFOLD = K * IN_CH / SIMD;
  const bool Overlap_Flag = (GUARD_BIT < 0);

  ap_uint<IPACK_BIT> ipacks[A_Sep][SIMD];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 2

  ap_int<M_BIT_Sep> PartialRes[2][Kp * PE][K + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 3

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
          if(Sep_Flag){
            KP_Pack_ACT_sep<IN_BIT, IN_BIT_H, IN_BIT_L, IPACK_BIT, AITV_BIT, Np, SIMD>(in_data, ipacks);
          }else{
            KP_Pack_ACT<IN_BIT, IPACK_BIT, AITV_BIT, Np, SIMD>(in_data, ipacks[0]);
          }
        }

        //shift and reset partial result accumulators
        if(flag_res_reset){
          for(unsigned k = 0; k < 2; k++){
            for(unsigned p = 0; p < Kp * PE; p++){
              for(unsigned i = 0; i < (K - 1); i++){
                PartialRes[k][p][i] = PartialRes[k][p][i + Np];
              }
              for(unsigned j = (K - 1); j < (K + Np - 1); j++){
                PartialRes[k][p][j] = 0;
              }
            }
          }   
        }

        for(unsigned p = 0; p < PE; p++){
          ap_int<ACC_BIT + 1> DSP_PartialRes[2][Kp][Np];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 3
          KP_Comp_SIMD_sep<IN_BIT, W_BIT,  W_BIT_H, W_BIT_L, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD, Np, Kp, CASCADE,
          WITV_BIT, ACC_BIT, W_Sep, A_Sep, Pattern_Flag, Overlap_Flag, Sep_Flag>(weights[p][och_offset + kich_counter], ipacks, DSP_PartialRes);

          for(unsigned k = 0; k < 2; k++){
            for(unsigned i = 0; i < Kp; i++){
              for(unsigned j = 0; j < Np; j++){
                PartialRes[k][i + p*Kp][kc_counter + j] += DSP_PartialRes[k][i][j];
              }
            }
          }
        }

        //output results
        if(flag_out){
          ap_int<Np * Kp * PE * M_BIT> out_data;
          for(unsigned p = 0; p < Kp * PE; p++){
            for(unsigned i = 0; i < Np; i++){
              ap_int<M_BIT> out_temp = (PartialRes[1][p][i] * (1 << ACC_Left_Shift)) + PartialRes[0][p][i];
              out_data(i*PE*Kp*M_BIT + p*M_BIT + M_BIT - 1, i*PE*Kp*M_BIT + p*M_BIT) = out_temp;
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

template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE, unsigned Kp,
          unsigned Np, unsigned CASCADE, int GUARD_BIT, unsigned M_BIT, unsigned SIMD_BIT,
          unsigned W_Sep, unsigned A_Sep, bool Pattern_Flag, unsigned kc_counter_bw,
          unsigned kich_counter_bw, unsigned och_offset_bw>
void KP_Array(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
              const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],
              stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
              const unsigned reps = 1){
  const unsigned SEL = W_Sep * A_Sep;

  if(SEL == 1){
    KP_Array_bas<K, ROW_LEN, IN_H, IN_CH, OUT_CH, IN_BIT, W_BIT, SIMD, PE, Kp, Np,
                 CASCADE, GUARD_BIT, M_BIT, SIMD_BIT, Pattern_Flag,
                 kc_counter_bw, kich_counter_bw, och_offset_bw>(in, weights, out, reps);
  }else{
    KP_Array_sep<K, ROW_LEN, IN_H, IN_CH, OUT_CH, IN_BIT, W_BIT, SIMD, PE, Kp, Np,
                 CASCADE, GUARD_BIT, M_BIT, SIMD_BIT, W_Sep, A_Sep, Pattern_Flag,
                 kc_counter_bw, kich_counter_bw, och_offset_bw>(in, weights, out, reps);
  }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

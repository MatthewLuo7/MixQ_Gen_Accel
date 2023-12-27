#ifndef __OPT_KP_HPP__
#define __OPT_KP_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"
#include "Opt_FP.hpp"



template <unsigned IN_BIT, unsigned IPACK_BIT, unsigned AITV_BIT, unsigned Np, unsigned SIMD>
void KP_Pack_ACT(ap_uint<Np * SIMD * IN_BIT> in_data,
                 ap_uint<IPACK_BIT> ipacks[SIMD]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  for (unsigned i = 0; i < SIMD; i++){
    ap_uint<IPACK_BIT> temp = 0;
    for (int j = 0; j < Np; j++){
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



template <unsigned IN_BIT, unsigned W_BIT, unsigned IPACK_BIT, unsigned WPACK_BIT,
          unsigned PROD_BIT, unsigned SIMD_BIT, unsigned SIMD, unsigned Np,
          unsigned Kp, unsigned CASCADE, bool Pattern_Flag>
void KP_Comp_SIMD_cascade(ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<WPACK_BIT> wpacks[SIMD],
                          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  ap_int<W_BIT + IN_BIT + SIMD_BIT> res[Kp * Np];
#pragma HLS ARRAY_PARTITION variable = res complete dim = 1

  for (unsigned t = 0; t < (Kp * Np); t++){
    res[t] = 0;
  }

  for (unsigned i = 0; i < SIMD; i += CASCADE){
    ap_int<PROD_BIT * Np * Kp> DSP_Res = 0;
    for (unsigned j = 0; j < CASCADE; j++){           //cascade through DSP's accumulator
      DSP_Res += ipacks[i+j] * wpacks[i+j];
    }

    ap_int<PROD_BIT> res0_temp = DSP_Res(PROD_BIT - 1, 0);
    res[0] += res0_temp;
    for (unsigned k = 1; k < (Kp * Np); k++){
      ap_int<PROD_BIT> res_temp = DSP_Res(k*PROD_BIT + PROD_BIT - 1, k*PROD_BIT);
      ap_int<PROD_BIT> acc_res_temp = res_temp + DSP_Res[k*PROD_BIT - 1];
      res[k] += acc_res_temp;
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
          unsigned Kp, bool Pattern_Flag>
void KP_Comp_SIMD_overlap(ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<WPACK_BIT> wpacks[SIMD],
                          ap_uint<Kp> wpfix[SIMD],
                          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  ap_int<W_BIT + IN_BIT + SIMD_BIT> res[Kp * Np];           //Initialize accumulation variables
#pragma HLS ARRAY_PARTITION variable = res complete dim = 1

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

    ap_int<PROD_BIT * Np * Kp + 1> DSP_Res = ipacks[i] * wpacks[i];                 //DSP packing multiplication

    ap_int<PROD_BIT + 1> rfix[Np*Kp];
#pragma HLS ARRAY_PARTITION variable = rfix complete

    rfix[0] = (fixsig[1], (ap_uint<PROD_BIT>) 0);
    rfix[Np*Kp - 1] = ((ap_uint<PROD_BIT>) 0, fixsig[Np*Kp - 1]^DSP_Res[(Np*Kp - 1)*PROD_BIT]);
    for (unsigned j = 1; j < (Np*Kp - 1); j++){
      rfix[j] = (fixsig[j+1], (ap_uint<PROD_BIT - 1>) 0, fixsig[j]^DSP_Res[j*PROD_BIT]);
    }

    for (unsigned k = 0; k < Np*Kp; k++){
      ap_int<PROD_BIT + 1> res_temp = DSP_Res((k+1)*PROD_BIT, k*PROD_BIT);
      ap_int<PROD_BIT + 1> acc_res_temp = res_temp + rfix[k];
      res[k] += acc_res_temp;
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
                  ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp][Np]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2

  if (Overlap_Flag){
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    ap_uint<Kp> wpfix[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
    KP_Pack_W_overlap<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks, wpfix);

    //SIMD computing array
    KP_Comp_SIMD_overlap<IN_BIT, W_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD, Np, Kp, Pattern_Flag>(ipacks, wpacks, wpfix, DSP_PartialRes);
  }else{
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    KP_Pack_W<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights, wpacks);

    //SIMD computing array
    KP_Comp_SIMD_cascade<IN_BIT, W_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD, Np, Kp, CASCADE, Pattern_Flag>(ipacks, wpacks, DSP_PartialRes);
  } 
}

template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE,
          unsigned Kp, unsigned Np, unsigned CASCADE, int GUARD_BIT,
          unsigned M_BIT, unsigned SIMD_BIT, unsigned adW_BIT, bool Pattern_Flag>
void KP_Array_Cascade(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                      const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],      // dim2: Kc --> Kr * IN_CH / SIMD --> OUT_CH / (Kp * PE)
                      stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
                      const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned PROD_BIT = IN_BIT + W_BIT + GUARD_BIT;
  const unsigned WITV_BIT = Pattern_Flag ? PROD_BIT : (Np * PROD_BIT);
  const unsigned AITV_BIT = Pattern_Flag ? (Kp * PROD_BIT) : PROD_BIT;
  const unsigned IPACK_BIT = (Np - 1) * AITV_BIT + IN_BIT;
  const unsigned WPACK_BIT = (Kp - 1) * WITV_BIT + W_BIT + adW_BIT;
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
  ap_uint<3> kc_counter = 0;
  ap_uint<12> kich_counter = 0;
  ap_uint<16> och_offset = 0;       //peIdx * K*INFOLD
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
          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp][Np];
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


template <unsigned K, unsigned IN_W, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned OUT_BIT, unsigned W_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned L_SHIFT, unsigned IN_PE, unsigned SIMD, unsigned PE, unsigned ACTP,
          unsigned Kp, unsigned Np, unsigned CASCADE, unsigned GUARD_BIT,
          unsigned M_BIT, unsigned SIMD_BIT, unsigned adW_BIT, bool Pattern_Flag>
void Conv_In_Wrapper(stream<ap_uint<IN_BIT * IN_CH> > &in,
                     const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],
                     const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                     const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                     stream<ap_uint<OUT_BIT * PE * Kp * 2> > &out, const unsigned reps = 1){

  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;

  stream<ap_uint<Np * SIMD * IN_BIT> > padding_out("padding_out");
  reshape_buffer_S2P<K, IN_H, IN_W, IN_CH, OUT_CH / PE, Np, IN_BIT, IN_PE, SIMD>(in, padding_out, reps);

  stream<ap_uint<Np * PE * Kp * M_BIT> > conv_out("conv_out");
  KP_Array_Cascade<K, ROW_LEN, IN_H, IN_CH, OUT_CH, IN_BIT, W_BIT, SIMD, PE, Kp, Np, CASCADE, GUARD_BIT, M_BIT, SIMD_BIT, adW_BIT, Pattern_Flag>(padding_out, weights, conv_out, reps);

  const unsigned convertnum_1 = IN_H * (OUT_CH / (PE * Kp)) * ROW_LEN;
  stream<ap_uint<ACTP * M_BIT> > convertnum_out("convertnum_out");
  StreamingDataWidthConverter_Batch<Np * PE * Kp * M_BIT, ACTP * M_BIT, convertnum_1>(conv_out, convertnum_out, reps);

  stream<ap_uint<ACTP * OUT_BIT> > ACT_out("ACT_out");
  Activation_Trim<K, IN_W, ROW_LEN, IN_H, OUT_CH, IN_BIT, OUT_BIT, W_BIT, INC_BIT, BIAS_BIT, L_SHIFT, PE, ACTP, Np, M_BIT>(convertnum_out, inc, bias, ACT_out, reps);

  const unsigned convertnum_2 = IN_H * (OUT_CH / (PE * Kp)) * IN_W * (PE * Kp / ACTP);
  StreamingDataWidthConverter_Batch<ACTP * OUT_BIT, 2 * PE * Kp * OUT_BIT, convertnum_2>(ACT_out, out, reps);
}


template <unsigned K, unsigned IN_W, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned OUT_BIT, unsigned W_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned L_SHIFT, unsigned IN_PE, unsigned SIMD, unsigned PE, unsigned ACTP,
          unsigned Kp, unsigned Np, unsigned CASCADE, unsigned GUARD_BIT,
          unsigned M_BIT, unsigned SIMD_BIT, unsigned adW_BIT, bool Pattern_Flag>
void Conv_In_Wrapper_KRowP(stream<ap_uint<IN_BIT * IN_CH> > &in,
                           const ap_uint<K * SIMD * Kp * W_BIT> weights[PE][K * (IN_CH / SIMD) * (OUT_CH / (Kp * PE))],
                           const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                           const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                           stream<ap_uint<OUT_BIT * PE * Kp * 2> > &out, const unsigned reps = 1){

  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;

  stream<ap_uint<Np * K * SIMD * IN_BIT> > padding_out("padding_out");
  reshape_buffer_FPT<K, IN_H, IN_W, IN_CH, OUT_CH / PE, Np, IN_BIT, IN_PE, SIMD>(in, padding_out, reps);

  stream<ap_uint<Np * PE * Kp * M_BIT> > conv_out("conv_out");
  KP_Array_Cascade<K, ROW_LEN, IN_H, IN_CH, OUT_CH, IN_BIT, W_BIT, SIMD * K, PE, Kp, Np, CASCADE, GUARD_BIT, M_BIT, SIMD_BIT, adW_BIT, Pattern_Flag>(padding_out, weights, conv_out, reps);

  const unsigned convertnum_1 = IN_H * (OUT_CH / (PE * Kp)) * ROW_LEN;
  stream<ap_uint<ACTP * M_BIT> > convertnum_out("convertnum_out");
  StreamingDataWidthConverter_Batch<Np * PE * Kp * M_BIT, ACTP * M_BIT, convertnum_1>(conv_out, convertnum_out, reps);

  stream<ap_uint<ACTP * OUT_BIT> > ACT_out("ACT_out");
  Activation_Trim<K, IN_W, ROW_LEN, IN_H, OUT_CH, IN_BIT, OUT_BIT, W_BIT, INC_BIT, BIAS_BIT, L_SHIFT, PE, ACTP, Np, M_BIT>(convertnum_out, inc, bias, ACT_out, reps);

  const unsigned convertnum_2 = IN_H * (OUT_CH / (PE * Kp)) * IN_W * (PE * Kp / ACTP);
  StreamingDataWidthConverter_Batch<ACTP * OUT_BIT, 2 * PE * Kp * OUT_BIT, convertnum_2>(ACT_out, out, reps);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

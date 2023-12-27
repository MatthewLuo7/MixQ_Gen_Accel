#ifndef __OPT_FP_HPP__
#define __OPT_FP_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"
#include "S2P_buffer.hpp"


template <unsigned IN_BIT, unsigned SIMD, unsigned Np, unsigned PROD_BIT, unsigned IPACK_BIT>
void FP_Pack_ACT(ap_uint<Np * SIMD * IN_BIT> in_data, ap_uint<IPACK_BIT> ipacks[SIMD]) {
#pragma HLS array_partition variable = ipacks

  for(unsigned i = 0; i < SIMD; i++){
    ap_uint<IPACK_BIT> temp = 0;
    for(unsigned j = 0; j < Np; j++){
      temp(j*PROD_BIT + IN_BIT - 1, j*PROD_BIT) = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
    }
    ipacks[i] = temp;
  }
}


template <unsigned W_BIT, unsigned SIMD, unsigned Kp, unsigned PROD_BIT, unsigned WPACK_BIT>
void FP_Pack_W(ap_uint<Kp * SIMD * W_BIT> in_weights, ap_int<WPACK_BIT> wpacks[SIMD]) {
#pragma HLS array_partition variable = wpacks

  for(unsigned i = 0; i < SIMD; i++) {
    ap_int<WPACK_BIT> wpack_temp = 0;
    for(unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = in_weights(j*SIMD*W_BIT + i*W_BIT + W_BIT - 1, j*SIMD*W_BIT + i*W_BIT);
      wpack_temp += (w_seg * (1 << (PROD_BIT * j)));
    }
    wpacks[i] = wpack_temp;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned Kp, unsigned PROD_BIT, unsigned WPACK_BIT>
void FP_Pack_W_overlap(ap_uint<Kp * SIMD * W_BIT> in_weights,
                       ap_int<WPACK_BIT> wpacks[SIMD],
                       ap_uint<Kp> wpfix[SIMD]) {
#pragma HLS array_partition variable = wpacks dim = 1
#pragma HLS array_partition variable = wpfix dim = 1

  for(unsigned i = 0; i < SIMD; i++) {
    ap_int<WPACK_BIT> wpack_temp = 0;
    for(unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = in_weights(j*SIMD*W_BIT + i*W_BIT + W_BIT - 1, j*SIMD*W_BIT + i*W_BIT);
      wpfix[i][j] = w_seg[0];
      wpack_temp += (w_seg * (1 << (PROD_BIT * j)));
    }
    wpacks[i] = wpack_temp;
  }
}


template <unsigned Kp, unsigned Np, unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT,
          unsigned SIMD, unsigned CASCADE, unsigned SIMD_BIT, unsigned WPACK_BIT, unsigned IPACK_BIT>
void FP_Comp_SIMD_cascade(ap_int<WPACK_BIT> wpacks[SIMD], ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp + Np - 1]) {
#pragma HLS ARRAY_PARTITION variable = wpacks complete
#pragma HLS ARRAY_PARTITION variable = ipacks complete
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete
  
  ap_int<W_BIT + IN_BIT + SIMD_BIT> rtemp[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = rtemp complete
  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    rtemp[i] = 0;
  }

  for(unsigned i = 0; i < SIMD; i += CASCADE) {
    ap_int<PROD_BIT * (Kp + Np - 1)> dspres = 0;
    for(unsigned cs = 0; cs < CASCADE; cs++) {
      dspres += wpacks[i + cs] * ipacks[i + cs];
    }

    ap_int<PROD_BIT> res_seg_0 = dspres(PROD_BIT - 1, 0);
    rtemp[0] += res_seg_0;
    for(unsigned j = 1; j < (Kp + Np - 1); j++){
      ap_int<PROD_BIT> res_seg = dspres(PROD_BIT*j + PROD_BIT - 1, PROD_BIT*j) + dspres[PROD_BIT*j - 1];
      rtemp[j] += res_seg;
    }
  }

  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    DSP_PartialRes[i] = rtemp[i];
  }
}

template <unsigned Kp, unsigned Np, unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT,
          unsigned SIMD, unsigned SIMD_BIT, unsigned WPACK_BIT, unsigned IPACK_BIT>
void FP_Comp_SIMD_overlap(ap_int<WPACK_BIT> wpacks[SIMD], ap_uint<Kp> wpfix[SIMD], ap_uint<IPACK_BIT> ipacks[SIMD],
                          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp + Np - 1]) {
#pragma HLS ARRAY_PARTITION variable = wpacks complete
#pragma HLS ARRAY_PARTITION variable = ipacks complete
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete
  
  ap_int<W_BIT + IN_BIT + SIMD_BIT> rtemp[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = rtemp complete
  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    rtemp[i] = 0;
  }

  for (unsigned i = 0; i < SIMD; i++){
    ap_uint<1> fixsig[Np + Kp - 1];
#pragma HLS ARRAY_PARTITION variable = fixsig complete dim = 1
    for(unsigned t = 0; t < (Np + Kp - 1); t++){
      fixsig[t] = 0;
    }

    for (unsigned Kp_i = 0; Kp_i < Kp; Kp_i++){                 //Calculate LSBs
      for (unsigned Np_i = 0; Np_i < Np; Np_i++){
        ap_uint<1> i_lsb = ipacks[i][PROD_BIT*Np_i];
        ap_uint<1> w_lsb = wpfix[i][Kp_i];
        fixsig[Kp_i + Np_i] ^= i_lsb&w_lsb;
      }
    }

    ap_int<PROD_BIT * (Kp + Np - 1) + 1> DSP_Res = ipacks[i] * wpacks[i];                 //DSP packing multiplication

    ap_int<PROD_BIT + 1> rfix[Np + Kp -1];
#pragma HLS ARRAY_PARTITION variable = rfix complete

    rfix[0] = (fixsig[1], (ap_uint<PROD_BIT>) 0);
    rfix[(Kp + Np - 1) - 1] = ((ap_uint<PROD_BIT>) 0, fixsig[(Kp + Np - 1) - 1]^DSP_Res[((Kp + Np - 1) - 1)*PROD_BIT]);
    for (unsigned j = 1; j < ((Kp + Np - 1) - 1); j++){
      rfix[j] = (fixsig[j+1], (ap_uint<PROD_BIT - 1>) 0, fixsig[j]^DSP_Res[j*PROD_BIT]);
    }

    for (unsigned k = 0; k < (Kp + Np - 1); k++){
      ap_int<PROD_BIT + 1> res_temp = DSP_Res((k+1)*PROD_BIT, k*PROD_BIT);
      ap_int<PROD_BIT + 1> acc_res_temp = res_temp + rfix[k];
      rtemp[k] += acc_res_temp;
    }
  }

  for(unsigned i = 0; i < (Kp + Np - 1); i++){
    DSP_PartialRes[i] = rtemp[i];
  }
}

template <unsigned Kp, unsigned Np, unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT,
          unsigned SIMD, unsigned CASCADE, unsigned SIMD_BIT, unsigned WPACK_BIT, unsigned IPACK_BIT, bool Overlap_Flag>
void FP_Comp_SIMD(ap_uint<Kp * SIMD * W_BIT> in_weights,
                  ap_uint<IPACK_BIT> ipacks[SIMD],
                  ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp + Np - 1]){
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1

  if(Overlap_Flag){
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    ap_uint<Kp> wpfix[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpfix complete dim = 1
    FP_Pack_W_overlap<W_BIT, SIMD, Kp, PROD_BIT, WPACK_BIT>(in_weights, wpacks, wpfix);

    //SIMD computing array
    FP_Comp_SIMD_overlap<Kp, Np, W_BIT, IN_BIT, PROD_BIT, SIMD, SIMD_BIT, WPACK_BIT, IPACK_BIT>(wpacks, wpfix, ipacks, DSP_PartialRes);
  }else{
    ap_int<WPACK_BIT> wpacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
    FP_Pack_W<W_BIT, SIMD, Kp, PROD_BIT, WPACK_BIT>(in_weights, wpacks);

    //SIMD computing array
    FP_Comp_SIMD_cascade<Kp, Np, W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, WPACK_BIT, IPACK_BIT>(wpacks, ipacks, DSP_PartialRes);
  }
}

/*
Dataflow: (SIMD * PE) * (Kp * Np) ---> ceil(K / Kp) ---> K * IN_CH / SIMD ---> ROW_LEN ---> OUTPENUM ---> IN_H
*/
template <unsigned K, unsigned ROW_LEN, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned W_BIT, unsigned SIMD, unsigned PE,
          unsigned Kp, unsigned Np, unsigned CASCADE, int GUARD_BIT,
          unsigned M_BIT, unsigned SIMD_BIT, unsigned adW_BIT>
void FP_Array_Cascade(
    stream<ap_uint<Np * SIMD * IN_BIT> > &in,
    const ap_uint<K * SIMD * W_BIT> weights[PE][(K * IN_CH / SIMD) * (OUT_CH / PE)],
    stream<ap_uint<Np * PE * M_BIT> > &out,
    const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  // static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  // static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  // static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");

  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = PROD_BIT * (Kp - 1) + W_BIT + adW_BIT;
  const unsigned IPACK_BIT = PROD_BIT * (Np - 1) + IN_BIT;
  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned INFOLD = K * IN_CH / SIMD;
  const unsigned KNUM = (K - 1) / Kp + 1;        //ceil(K / Kp) 
  const bool Overlap_Flag = ((2 << GUARD_BIT) < Kp) && ((2 << GUARD_BIT) < Np);

  ap_uint<IPACK_BIT> ipacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  ap_int<M_BIT> PartialRes[PE][Kp * KNUM + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;
  ap_uint<K * SIMD * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1

  //counters
  ap_uint<3> k_counter = 0;
  ap_uint<12> infold_counter = 0;
  ap_uint<5> res_offset = 0;
  ap_uint<16> add_offset = 0;       //peIdx * INFOLD

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
          FP_Pack_ACT<IN_BIT, SIMD, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks);
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

          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp + Np - 1];
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          FP_Comp_SIMD<Kp, Np, W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, WPACK_BIT, IPACK_BIT, Overlap_Flag>(in_weights, ipacks, DSP_PartialRes);

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

/*
Constraints

K:
K <= IN_W
K = 1, 3, 5, 7

IN_W, IN_H:
K <= IN_W <= 512
K <= IN_H <= 512

IN_CH, OUT_CH:
1 <= IN_CH <= 256
1 <= OUT_CH <= 256

IN_BIT, OUT_BIT, W_BIT:
2 <= IN_BIT <= 8
2 <= OUT_BIT <= 8
2 <= W_BIT <= 8

INC_BIT, BIAS_BIT:

IN_PE, SIMD, PE, ACTP:
IN_PE % SIMD == 0, if IN_PE > SIMD
SIMD % IN_PE 7== 0, if SIMD >= IN_PE
SIMD <= IN_CH
IN_CH % SIMD == 0
PE <= OUT_CH
OUT_CH % PE == 0
PE % ACTP == 0

Kp, Np, CASCADE, GUARD_BIT
1 <= Kp <= K
1 <= Np <= IN_W
SIMD % CADCADE == 0
1 <= CASCADE <= floor((2**GUARD_BIT) / min(Kp, Np))

M_BIT, SIMD_BIT:
M_BIT == IN_BIT + W_BIT + ceil(log2(K * K * IN_CH))
SIMD_BIT == ceil(log2(SIMD * min(Kp, Np)))
*/


template <unsigned K, unsigned IN_W, unsigned IN_H, unsigned IN_CH, unsigned OUT_CH,
          unsigned IN_BIT, unsigned OUT_BIT, unsigned W_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned L_SHIFT, unsigned IN_PE, unsigned SIMD, unsigned PE, unsigned ACTP,
          unsigned Kp, unsigned Np, unsigned CASCADE, int GUARD_BIT,
          unsigned M_BIT, unsigned SIMD_BIT, unsigned adW_BIT>
void Conv_Opt_Test_Wrapper(
    stream<ap_uint<2 * IN_PE * IN_BIT> > &in,
    const ap_uint<K * SIMD * W_BIT> weights[PE][(K * IN_CH / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[ACTP][OUT_CH / ACTP],
    const ap_int<BIAS_BIT> bias[ACTP][OUT_CH / ACTP],
    stream<ap_uint<PE * 2 * OUT_BIT> > &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;

  const unsigned convertnum_0 = IN_H * (IN_CH / IN_PE) * (IN_W / 2);
  stream<ap_uint<IN_PE * IN_BIT> > convertnum_out_0("convertnum_out_0");
  StreamingDataWidthConverter_Batch<2 * IN_PE * IN_BIT, IN_PE * IN_BIT, convertnum_0>(in, convertnum_out_0, reps);

  stream<ap_uint<Np * SIMD * IN_BIT> > padding_out("padding_out");
  reshape_buffer_S2P<K, IN_H, IN_W, IN_CH, OUT_CH / PE, Np, IN_BIT, IN_PE, SIMD>(convertnum_out_0, padding_out, reps);

  stream<ap_uint<Np * PE * M_BIT> > conv_out("conv_out");
  FP_Array_Cascade<K, ROW_LEN, IN_H, IN_CH, OUT_CH, IN_BIT, W_BIT, SIMD, PE, Kp, Np, CASCADE, GUARD_BIT, M_BIT, SIMD_BIT, adW_BIT>(padding_out, weights, conv_out, reps);

  const unsigned convertnum_1 = IN_H * (OUT_CH / PE) * ROW_LEN;
  stream<ap_uint<ACTP * M_BIT> > convertnum_out("convertnum_out");
  StreamingDataWidthConverter_Batch<Np * PE * M_BIT, ACTP * M_BIT, convertnum_1>(conv_out, convertnum_out, reps);

  stream<ap_uint<ACTP * OUT_BIT> > ACT_out("ACT_out");
  Activation_Trim<K, IN_W, ROW_LEN, IN_H, OUT_CH, IN_BIT, OUT_BIT, W_BIT, INC_BIT, BIAS_BIT, L_SHIFT, PE, ACTP, Np, M_BIT>(convertnum_out, inc, bias, ACT_out, reps);

  const unsigned convertnum_2 = IN_H * (OUT_CH / PE) * IN_W * (PE / ACTP);
  StreamingDataWidthConverter_Batch<ACTP * OUT_BIT, 2 * PE * OUT_BIT, convertnum_2>(ACT_out, out, reps);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

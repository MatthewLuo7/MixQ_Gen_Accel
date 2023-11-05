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
    for (int j = 0; j < Np - 1; j++){
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


template <unsigned IN_BIT, unsigned W_BIT, unsigned IPACK_BIT, unsigned WPACK_BIT,
          unsigned PROD_BIT, unsigned SIMD_BIT, unsigned SIMD, unsigned Np,
          unsigned Kp, unsigned CASCADE, bool Pattern1>
void KP_Comp_SIMD(ap_uint<IPACK_BIT> ipacks[SIMD],
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
  if(Pattern1){                             //weights are packed more densely
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




template <unsigned IN_H, unsigned IN_CH, unsigned OUT_CH, unsigned IN_BIT, unsigned W_BIT,
          unsigned SIMD, unsigned Np, unsigned Kp, unsigned PE, unsigned OCH_FOLD_num,
          unsigned Np_num, unsigned SIMD_num, unsigned M_BIT, unsigned SIMD_BIT,
          unsigned GUARD_BIT, unsigned EXWPACK_BIT, unsigned CASCADE, bool Pattern1>
void FP_Array_Cascade(stream<ap_uint<Np * SIMD * IN_BIT> > &in,
                      const ap_uint<SIMD * Kp * W_BIT> weights[PE][K * (K * IN_CH / SIMD) * (OUT_CH / (Kp * PE))],      // dim2: Kc --> Kr * IN_CH / SIMD --> OUT_CH / (Kp * PE)
                      stream<ap_uint<Np * PE * Kp * M_BIT> > &out,
                      const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  const unsigned PROD_BIT = IN_BIT + W_BIT + GUARD_BIT;
  const unsigned WITV_BIT = Pattern1 ? PROD_BIT : (Np * PROD_BIT);
  const unsigned AITV_BIT = Pattern1 ? (Kp * PROD_BIT) : PROD_BIT;
  const unsigned IPACK_BIT = (Np - 1) * AITV_BIT + IN_BIT;
  const unsigned WPACK_BIT = (Kp - 1) * WITV_BIT + W_BIT + EXWPACK_BIT;
  const unsigned OUTPENUM = OUT_CH / (PE * Kp);
  const unsigned INFOLD = K * IN_CH / SIMD;

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  ap_int<M_BIT> PartialRes[Kp * PE][K + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_int<M_BIT> conv_acc[PE][Kp * Np];
#pragma HLS ARRAY_PARTITION variable = conv_acc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_acc complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;

  //counters
  ap_uint<3> kr_counter = 0;
  ap_uint<12> kich_counter = 0;
  ap_uint<16> och_offset = 0;       //peIdx * K*INFOLD
  for (unsigned h = 0; h < IN_H * reps; h++){
    for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for (unsigned cycle = 0; cycle < K * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline II = 1

        //flags for input, result reset, and output
        bool flag_res_reset = (kich_counter == 0);
        bool flag_out = (kich_counter == (K*INFOLD - 1));

        //input and pack activations
        in_data = in.read();
        KP_Pack_ACT<IN_BIT, IPACK_BIT, AITV_BIT, Np, SIMD>(in_data, ipacks);

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
          //load and pack weights
          KP_Pack_W<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights[p][och_offset + kich_counter], wpacks[p]);

          //SIMD computing array
          ap_int<W_BIT + IN_BIT + SIMD_BIT> DSP_PartialRes[Kp][Np];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 2
          KP_Comp_SIMD<IN_BIT, W_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD, Np, Kp, CASCADE, Pattern1>(wpacks[p], ipacks, DSP_PartialRes);

          for(unsigned i = 0; i < Kp; i++){
            for(unsigned j = 0; j < Np; j++){
              PartialRes[i + p*Kp][kr_counter + j] += DSP_PartialRes[i][j];
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
        kr_counter++;
        if(kr_counter == K){
          k_counter = 0;
        }

        kich_offset++;
        if(kich_offset == K*INFOLD){
          kich_offset = 0
        }

        if(cycle == (K*INFOLD*ROW_LEN - 1)){
          och_offset += K*INFOLD;
          if(och_offset == OUTPENUM*K*INFOLD){
            och_offset = 0;
          }
        }


/////////////////////////////////////////////////////////////////////////////////////////

//         bool acc_reset = (SIMD_num_count == 0);
//         bool acc_output = (SIMD_num_count == SIMD_num - 1);

//         ap_uint<IN_BIT * Np * SIMD> inData;
//         inData = in.read();
//         Conv1x1_Input_Pack<IN_BIT, AITV_BIT, IPACK_BIT, Np, SIMD>(inData, ipacks);

//         for (unsigned p = 0; p < PE; p++){
// #pragma HLS unroll
//           ap_int<WPACK_BIT> wpacks[SIMD];
// #pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
//           Conv1x1_Weight_Pack<W_BIT, WITV_BIT, WPACK_BIT, Kp, SIMD>(weights[p][ofIdx*SIMD_num + SIMD_num_count], wpacks); 

//           ap_int<W_BIT + IN_BIT + SIMD_BIT> conv_acc_SIMD[Kp * Np];
// #pragma HLS ARRAY_PARTITION variable = conv_acc_SIMD complete dim = 1
//           Conv1x1_DSPMul_cascade<IN_BIT, W_BIT, IPACK_BIT, WPACK_BIT, PROD_BIT, SIMD_BIT, SIMD, Np, Kp, CASCADE>(ipacks, wpacks, conv_acc_SIMD);

//           for (unsigned i = 0; i < Kp * Np; i++){
// #pragma HLS unroll
//             if (acc_reset){
//               conv_acc[p][i] = conv_acc_SIMD[i];
//             }
//             else{
//               conv_acc[p][i] += conv_acc_SIMD[i];
//             }
//           }
//         }

//         if (acc_output){
//           ap_uint<M_BIT * Kp * PE * Np> outData;

//           for (unsigned i = 0; i < Np; i++){
// #pragma HLS unroll
//             for (unsigned j = 0; j < PE; j++){
// #pragma HLS unroll
//               for (unsigned k = 0; k < Kp; k++){
// #pragma HLS unroll
//                 //reorder sequence
//                 if (WITV_BIT < AITV_BIT){     //Kp weights are packed on 18-bit port
//                   outData(k*M_BIT + j*M_BIT*Kp + i*M_BIT*Kp*PE + M_BIT - 1, k*M_BIT + j*M_BIT*Kp + i*M_BIT*Kp*PE) = conv_acc[j][k + i*Kp];
//                 }
//                 else{                         //Np activations are packed on 18-bit port
//                   outData(k*M_BIT + j*M_BIT*Kp + i*M_BIT*Kp*PE + M_BIT - 1, k*M_BIT + j*M_BIT*Kp + i*M_BIT*Kp*PE) = conv_acc[j][i + k*Np];
//                 }
//               }
//             }
//           }
//           out.write(outData);
//         }

//         SIMD_num_count += 1;
//         if (SIMD_num_count == SIMD_num){
//           SIMD_num_count = 0;
//         }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

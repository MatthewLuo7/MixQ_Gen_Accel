#ifndef __CONV_OPT_HPP__
#define __CONV_OPT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"


template <unsigned IN_BIT, unsigned SIMD, unsigned Np, unsigned PROD_BIT, unsigned IPACK_BIT>
void Conv_Pack_ACT(ap_uint<Np * SIMD * IN_BIT> in_data, ap_uint<IPACK_BIT> ipacks[SIMD]) {
#pragma HLS array_partition variable = ipacks

  for(unsigned i = 0; i < SIMD; i++){
    ap_uint<IPACK_BIT> temp = 0;
    for(int j = 0; j < Np; j++){
      temp(j*PROD_BIT + IN_BIT - 1, j*PROD_BIT) = in_data(j*SIMD*IN_BIT + i*IN_BIT + IN_BIT - 1, j*SIMD*IN_BIT + i*IN_BIT);
    }
    ipacks[i] = temp;
  }
}


template <unsigned W_BIT, unsigned SIMD, unsigned Kp, unsigned PROD_BIT, unsigned WPACK_BIT>
void Conv_Pack_W(ap_uint<Kp * SIMD * W_BIT> in_weights, ap_int<WPACK_BIT> wpacks[SIMD]) {
#pragma HLS array_partition variable = wpacks

  for(unsigned i = 0; i < SIMD; i++) {
    wpacks[i] = 0;
    for(unsigned j = 0; j < Kp; j++){
      ap_int<W_BIT> w_seg = in_weights(j*SIMD*W_BIT + i*W_BIT + W_BIT - 1, j*SIMD*W_BIT + i*W_BIT);
      wpacks[i] += (w0_seg * (1 << (PROD_BIT * (Kp - j - 1))));
    }
  }
}


template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD, unsigned CASCADE,
          unsigned SIMD_BIT, unsigned PA_BIT, unsigned WPACK_BIT, unsigned IPACK_BIT>
void Conv_Comp_SIMD(ap_int<WPACK_BIT> wpacks[SIMD], ap_uint<IPACK_BIT> ipacks[SIMD], ap_int<W_BIT + IN_BIT + SIMD_BIT + PA_BIT> DSP_PartialRes[Kp + Np - 1]) {
#pragma HLS ARRAY_PARTITION variable = wpacks complete
#pragma HLS ARRAY_PARTITION variable = ipacks complete
#pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete
  
  ap_int<W_BIT + IN_BIT + SIMD_BIT + PA_BIT> rtemp[Kp + Np - 1];

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


/*
Dataflow: (SIMD * PE) * (Kp * Np) ---> ceil(K / Kp) ---> K * IN_CH / SIMD ---> ROW_LEN ---> OUTPENUM ---> OUT_H
*/
template <unsigned K, unsigned Kp, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_W,
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT, unsigned GUARD_BIT,
          unsigned M_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE, 
          unsigned SIMD_BIT, unsigned PA_BIT, unsigned adW_BIT>
void Conv_cascade(
    stream<ap_uint<Np * SIMD * IN_BIT> > &in,
    const ap_uint<K * SIMD * W_BIT> weights[PE][(K * IN_CH / SIMD) * (OUT_CH / PE)],
    stream<ap_uint<Np * PE * M_BIT> > &out,
    const unsigned reps = 1) {

  //static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  //static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  //static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");

  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = PROD_BIT * (Kp - 1) + W_BIT + adW_BIT;
  const unsigned IPACK_BIT = PROD_BIT * (Np - 1) + IN_BIT;
  const unsigned OUTPENUM = OUT_CH / PE;
  const unsigned INFOLD = K * IN_CH / SIMD;
  const unsigned KNUM = (K - 1) / Np;        //ceil(K / Kp) 

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipacks[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipacks complete dim = 1

  ap_int<M_BIT> PartialRes[PE][K + Np - 1];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;
  ap_uint<K * SIMD * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1

  //counters
  ap_uint<3> k_counter = 0;
  ap_uint<11> infold_counter = 0;
  ap_uint<8> res_offset = 0;

  for(unsigned h = 0; h < OUT_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < KNUM * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline

        //flags for input, result reset, and output
        bool flag_in = (k_counter == 0);
        bool flag_res_reset = ((infold_counter == 0) && (k_counter == 0));
        bool flag_out = ((infold_counter == (INFOLD - 1)) && (k_counter == (KNUM - 1)));

        //input new activations and load weights
        if(flag_in){
          in_data = in.read();
          Conv_Pack_ACT<IN_BIT, SIMD, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks);
          for(unsigned p = 0; p < PE; p++){
            cur_weights[p] = weights[p][peIdx * INFOLD + infold_counter];
          }
        }

        //computing array, PE * SIMD array
        for(unsigned p = 0; p < PE; p++){
          //extract Kp weights
          ap_uint<Kp * SIMD * W_BIT> in_weights = cur_weights[p](Kp * SIMD * W_BIT - 1, 0);
          cur_weights[p] = cur_weights[p] >> (SIMD * W_BIT);
          Conv_Pack_W<W_BIT, SIMD, Kp, PROD_BIT, WPACK_BIT>(in_weights, wpacks[p]);

          //SIMD computing array
          ap_int<W_BIT + IN_BIT + SIMD_BIT + PA_BIT> DSP_PartialRes[Kp + Np - 1];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1
          Conv_Comp_SIMD<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, PA_BIT, WPACK_BIT, IPACK_BIT>(wpacks[p], ipacks, DSP_PartialRes);

          //shift and reset partial result accumulators
          if(flag_res_reset){
            for(unsigned i = 0; i < (K - 1); i++){
              PartialRes[p][i] = PartialRes[p][i + Np];
            }
            for(unsigned j = (K - 1); j < (K + Np - 1); j++){
              PartialRes[p][j] = 0;
            }
          }

          //accumulate partial results
          for(unsigned i = 0; (i < (Kp + Np - 1)) && (i < (K + Np - 1 - res_offset)); i++){
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
        res_offset += Np;
        if(k_counter == KNUM){
          k_counter = 0;
          res_offset = 0;
          infold_counter++;
          if(infold_counter == INFOLD){
            infold_counter = 0;
          }
        }
      }
    }
  }
}

//-------------------------------------------------------------------------convolution dataflow-----------------------------------------------------------

template <unsigned IN_ROW, unsigned IN_COL, unsigned OUT_CH, unsigned PE,
          unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT, unsigned IN_BIT,
          unsigned OUT_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned ACT_SIMD>
void  conv3x3_1Dopt_conv_ACT( stream<ap_uint<M_BIT * ACT_SIMD> > &in,
                const ap_int<INC_BIT> inc[ACT_SIMD][OUT_CH / ACT_SIMD],
                const ap_int<BIAS_BIT> bias[ACT_SIMD][OUT_CH / ACT_SIMD],
                stream<ap_uint<OUT_BIT * ACT_SIMD> > &out,
                const unsigned reps = 1 ){
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
const unsigned ACT_num = PE / ACT_SIMD;

  unsigned ACT_num_count = 0;
  for (unsigned int h = 0; h < IN_ROW * reps; h++) 
  {
    for (unsigned int peIdx = 0; peIdx < (OUT_CH / PE); peIdx++)
    {
      for (unsigned int i = 0; i < (ACT_num * IN_COL); i++)
      {
      #pragma HLS pipeline

        ap_int<M_BIT * ACT_SIMD> iData;
        iData = in.read();
        ap_uint<OUT_BIT * ACT_SIMD> oData;
        for (unsigned int j = 0; j < ACT_SIMD; j++)
        {
        #pragma HLS unroll

          oData((j + 1) * OUT_BIT - 1, j * OUT_BIT) = bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>
          (iData((j + 1) * M_BIT - 1, j * M_BIT), inc[j][ACT_num_count + peIdx*ACT_num], bias[j][ACT_num_count + peIdx*ACT_num]);
        }
        out.write(oData);

        ACT_num_count += 1;
        if (ACT_num_count == ACT_num){
          ACT_num_count = 0;
        }
      }
    }
  }
}


template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned OUT_CH, unsigned OUT_BIT,unsigned W_BIT, unsigned M_BIT,
          unsigned INC_BIT, unsigned BIAS_BIT,unsigned SIMD, unsigned CASCADE,
          unsigned IN_PE, unsigned PE, unsigned L_SHIFT, unsigned ACT_SIMD,
          unsigned SIMD_BIT, unsigned adW_BIT, unsigned GUARD_BIT>
void conv3x3_1Dopt_2a_cascade(
    stream<ap_uint<IN_BIT * IN_PE * 2> > &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3][((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[ACT_SIMD][OUT_CH / ACT_SIMD],
    const ap_int<BIAS_BIT> bias[ACT_SIMD][OUT_CH / ACT_SIMD],
    stream<ap_uint<OUT_BIT * PE * 2> > &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  stream<ap_uint<SIMD * IN_BIT * 2> > padding_out("padding_out");
  conv3x3_1Dopt_2a_padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(in, padding_out, reps);

  stream<ap_uint<PE * M_BIT * 2> > conv_out("conv_out");
  conv3x3_1Dopt_2a_array_cascade<3, IN_BIT, IN_CH, OUT_COL, OUT_ROW, OUT_CH, W_BIT, GUARD_BIT, M_BIT, SIMD, CASCADE, PE, SIMD_BIT, adW_BIT>(padding_out, weights, conv_out, reps);

  const unsigned convertnum_1 = OUT_ROW * (OUT_CH / PE) * (OUT_COL / 2);
  stream<ap_uint<M_BIT * ACT_SIMD> > convertnum_out("convertnum_out");
  StreamingDataWidthConverter_Batch<PE * M_BIT * 2, M_BIT * ACT_SIMD, convertnum_1>(conv_out, convertnum_out, reps);

  stream<ap_uint<OUT_BIT * ACT_SIMD> > ACT_out("ACT_out");
  conv3x3_1Dopt_conv_ACT<IN_ROW, IN_COL, OUT_CH, PE, M_BIT, INC_BIT, BIAS_BIT, IN_BIT, OUT_BIT, W_BIT, L_SHIFT, ACT_SIMD>(convertnum_out, inc, bias, ACT_out, reps);

  const unsigned convertnum_2 = convertnum_1 * 2 * PE / ACT_SIMD;
  StreamingDataWidthConverter_Batch<OUT_BIT * ACT_SIMD, PE * OUT_BIT * 2, convertnum_2>(ACT_out, out, reps);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------

#endif

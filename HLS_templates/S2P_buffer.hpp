#ifndef __CONV3X3_1DOPT_2A_HPP__
#define __CONV3X3_1DOPT_2A_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "stream_tools.h"


//-----------------------------------------------------------------------padding and sliding window--------------------------------------------------------------------



template <unsigned K, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
 unsigned SIMD, unsigned Np, unsigned ROW_LEN>
void stream_in_row_SIMD_INPE(
    stream<ap_uint<IN_PE * IN_BIT> > &in,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[SIMD / IN_PE][K + 1][ROW_LEN * (IN_CH / SIMD)],
    bool skip_flag, ap_uint<3> rowBufferIdx){
  const unsigned PAD_LEN = (K - 1) / 2;
#pragma HLS inline off

  if (skip_flag){
    return;
  }

  ap_uint<9> l_c = 0;
  ap_uint<4> n_c = 0;
  ap_uint<IN_PE * IN_BIT * Np> reg = 0;
  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (unsigned w_counter = 0; w_counter < ROW_LEN * Np; w_counter++){
#pragma HLS pipeline
        reg = reg >> (IN_PE * IN_BIT);

        ap_uint<IN_PE * IN_BIT> data;
        if ((w_counter < PAD_LEN) || ( w_counter >= (PAD_LEN + IN_W))) {
          data = 0;
        } else {
          data = in.read();
        }

        reg(IN_PE * IN_BIT * Np - 1, IN_PE * IN_BIT * (Np - 1)) = data;
        n_c++;
        if (n_c == Np){
          n_c = 0;
          row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [l_c * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
          l_c++；
          if (l_c == ROW_LEN){
            l_c = 0;
          }
        }
    }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_rows_SIMD_INPE(
    stream<ap_uint<SIMD * IN_BIT * 2> > &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [(IN_W / 2 + 1) * IN_CH / SIMD],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2 + 1;
  if (skip_flag)
    return;

  ap_uint<8> infoldIdx = 0;
  ap_uint<8> w = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < WLEN * K * SIMDNUM; cycle++) {
      // for (unsigned w = 0; w < WLEN; w++) {
      //   for (unsigned wr = 0; wr < K; wr++) {
      //     for (unsigned simdIdx = 0; simdIdx < SIMDNUM; simdIdx++) {
      ap_uint<2> wr = infoldIdx / SIMDNUM;
      ap_uint<4> simdIdx = infoldIdx % SIMDNUM;
#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0;
      ap_uint<SIMD * IN_BIT> data1;
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[SIMD / IN_PE];
#pragma HLS array_partition variable = buffer_data complete
      ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
      for (unsigned i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
        buffer_data[i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];
      }

      if (outRowIdx - K / 2 + wr < 0 || outRowIdx - K / 2 + wr >= IN_H) {
        data0 = 0;
        data1 = 0;
      } else {
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          data0((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT - 1, 0);
          data1((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
        }
      }
      out.write((data1, data0));

      if (cycle == WLEN * K * SIMDNUM - 1) {
        w = 0;
      } else if (infoldIdx == K * SIMDNUM - 1) {
        w++;
      }

      if (infoldIdx == K * SIMDNUM - 1) {
        infoldIdx = 0;
      } else {
        infoldIdx++;
      }
    }
  }
}




/*
K = 1, 3, 5, 7
IN_W < 512
*/
template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned OUT_CH,
          unsigned Kp, unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned SIMD>
void reshape_buffer_SIMD_INPE(stream<ap_uint<IN_PE * IN_BIT> > &in,
                              stream<ap_uint<SIMD * IN_BIT * Np> > &out,
                              const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)

  ap_uint<IN_PE * IN_BIT * Np> row_buffer[SIMD / IN_PE][K + 1][ROW_LEN * (IN_CH / SIMD)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<3> storeBufferIdx = 0;
  ap_uint<3> loadBufferIdx = 1;
  ap_int<10> rowIdx = - (K - 1);

  for (unsigned rep = 0; rep < reps * IN_H + (K - 1); rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row_SIMD_INPE<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    stream_out_rows_SIMD_INPE<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(out, row_buffer, (rep < (K - 1)), rowIdx, loadBufferIdx);
    loadBufferIdx++;
    if (loadBufferIdx == (K + 1)){
      loadBufferIdx -= (K + 1);
    }
    storeBufferIdx++;
    if (storeBufferIdx == (K + 1)){
      storeBufferIdx -= (K + 1);
    }

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------


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

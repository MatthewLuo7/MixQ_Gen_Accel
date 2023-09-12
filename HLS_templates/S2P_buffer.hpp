#ifndef __S2P_BUFFER_HPP__
#define __S2P_BUFFER_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"


//-----------------------------------------------------------------------padding and sliding window--------------------------------------------------------------------
template <unsigned K, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD, unsigned Np, unsigned ROW_LEN>
void stream_in_row_SIMD_INPE(
    stream<ap_uint<IN_PE * IN_BIT> > &in,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[SIMD / IN_PE][K + 1][ROW_LEN * (IN_CH / SIMD)],
    bool skip_flag, ap_uint<3> rowBufferIdx){
#pragma HLS inline off
  const unsigned PAD_LEN = (K - 1) / 2;

  if (skip_flag)
    return;

  //counters
  ap_uint<4> n_c = 0;
  ap_uint<5> simd_ipe_c = 0;
  ap_uint<5> ch_simd_c = 0;
  ap_uint<14> mem_offset = 0;

  ap_uint<IN_PE * IN_BIT * Np> reg = 0;
  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++){
    for (unsigned w_counter = 0; w_counter < ROW_LEN * Np; w_counter++){
#pragma HLS pipeline
      reg = reg >> (IN_PE * IN_BIT);

      ap_uint<IN_PE * IN_BIT> data;
      if ((w_counter < PAD_LEN) || (w_counter >= (PAD_LEN + IN_W))) {
        data = 0;
      } else {
        data = in.read();
      }

      reg(IN_PE * IN_BIT * Np - 1, IN_PE * IN_BIT * (Np - 1)) = data;

      //counters
      n_c++;
      if(n_c == Np){
        n_c = 0;
        row_buffer[simd_ipe_c][rowBufferIdx][mem_offset + ch_simd_c] = reg;
        mem_offset += (IN_CH / SIMD);
        if(mem_offset == ROW_LEN * IN_CH / SIMD){
          // rl_c = 0;
          mem_offset = 0;
          simd_ipe_c++;
          if(simd_ipe_c == (SIMD / IN_PE)){
            simd_ipe_c = 0;
            ch_simd_c++;
            if(ch_simd_c == (IN_CH / SIMD)){
              ch_simd_c = 0;
            }
          }
        }
      }
    }
  }
}


template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned Np, unsigned ROW_LEN, unsigned OUTPENUM>
void stream_out_rows_SIMD_INPE(
    stream<ap_uint<SIMD * IN_BIT * Np> > &out,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[SIMD / IN_PE][K + 1][ROW_LEN * (IN_CH / SIMD)],
    bool skip_flag, ap_int<10> outRowIdx, ap_uint<3> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMD_BIT = SIMD * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;

  if (skip_flag)
    return;

  //counters
  ap_uint<3> kr_c = 0;
  ap_uint<5> simd_c = 0;
  ap_uint<14> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < ROW_LEN * K * SIMDNUM; cycle++) {
#pragma HLS pipeline

      ap_uint<SIMD * IN_BIT * Np> write_data;
      ap_uint<SIMD * IN_BIT> data[Np];
#pragma HLS array_partition variable = data complete
      ap_uint<IN_PE * IN_BIT * Np> buffer_data[SIMD / IN_PE];
#pragma HLS array_partition variable = buffer_data complete

      ap_uint<3> rowBufferIdx = startRowBufferIdx + kr_c;
      if (rowBufferIdx >= (K + 1)){
        rowBufferIdx -= (K + 1);
      }
      for (unsigned i = 0; i < SIMD / IN_PE; i++) {
        buffer_data[i] = row_buffer[i][rowBufferIdx][mem_offset + simd_c];
      }

      if ((outRowIdx - (K / 2) + kr_c < 0) || (outRowIdx - (K / 2) + kr_c >= IN_H)) {
        write_data = 0;
      } else {
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          for (unsigned j = 0; j < Np; j++){
            data[j]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[i]((j+1)*IN_PE_BIT - 1, j*IN_PE_BIT);
          }
        }

        for (unsigned j = 0; j < Np; j++){
          write_data((j+1)*SIMD_BIT - 1, j*SIMD_BIT) = data[j];
        }
      }
      out.write(write_data);

      //counters
      simd_c++;
      if (simd_c == SIMDNUM){
        simd_c = 0;
        kr_c++;
        if (kr_c == K){
          kr_c = 0;
          mem_offset += SIMDNUM;
          if (mem_offset == ROW_LEN * SIMDNUM){
            mem_offset = 0;
          }
        }
      }
    }
  }
}




/*
K = 1, 3, 5, 7
IN_W < 512
*/
template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned OUTPENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned SIMD>
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
    stream_in_row_SIMD_INPE<K, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, Np, ROW_LEN>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    stream_out_rows_SIMD_INPE<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, Np, ROW_LEN, OUTPENUM>(out, row_buffer, (rep < (K - 1)), rowIdx, loadBufferIdx);
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


template <unsigned K, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD, unsigned Np, unsigned ROW_LEN>
void stream_in_row_INPE_SIMD(
    stream<ap_uint<IN_PE * IN_BIT> > &in,
    ap_uint<SIMD * IN_BIT * Np> row_buffer[IN_PE / SIMD][K + 1][ROW_LEN * (IN_CH / IN_PE)],
    bool skip_flag, ap_uint<3> rowBufferIdx){
#pragma HLS inline off
  const unsigned PAD_LEN = (K - 1) / 2;

  if (skip_flag)
    return;

  ap_uint<14> mem_offset = 0;
  ap_uint<4> n_c = 0;
  ap_uint<SIMD * IN_BIT * Np> reg[IN_PE / SIMD];
#pragma HLS ARRAY_PARTITION variable = reg dim = 1 complete
  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++){
    for (unsigned w_counter = 0; w_counter < ROW_LEN * Np; w_counter++){
#pragma HLS pipeline
      for (unsigned i = 0; i < (IN_PE / SIMD); i++){
        reg[i] = reg[i] >> (SIMD * IN_BIT);
      }
      
      ap_uint<IN_PE * IN_BIT> data;
      if ((w_counter < PAD_LEN) || (w_counter >= (PAD_LEN + IN_W))) {
        data = 0;
      } else {
        data = in.read();
      }

      for (unsigned i = 0; i < (IN_PE / SIMD); i++){
        reg[i](SIMD*IN_BIT*Np - 1, SIMD*IN_BIT*(Np-1)) = data(SIMD*IN_BIT*(i+1) - 1, SIMD*IN_BIT*i);
      }
      
      n_c++;
      if (n_c == Np){
        n_c = 0;

        for (unsigned i = 0; i < (IN_PE / SIMD); i++){
          row_buffer[i][rowBufferIdx][mem_offset + peIdx] = reg[i];
        }

        mem_offset += (IN_CH / IN_PE);
        if (mem_offset == ROW_LEN * IN_CH / IN_PE){
          mem_offset = 0;
        }
      }
    }
  }
}



template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned Np, unsigned ROW_LEN, unsigned OUTPENUM>
void stream_out_rows_INPE_SIMD(
    stream<ap_uint<SIMD * IN_BIT * Np> > &out,
    ap_uint<SIMD * IN_BIT * Np> row_buffer[IN_PE / SIMD][K + 1][ROW_LEN * (IN_CH / IN_PE)],
    bool skip_flag, ap_int<10> outRowIdx, ap_uint<3> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMD_BIT = SIMD * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned INPENUM = IN_CH / IN_PE;

  if (skip_flag)
    return;

  //counters
  ap_uint<3> kr_c = 0;
  ap_uint<5> ch_ipe_c = 0;
  ap_uint<5> ipe_simd_c = 0;
  ap_uint<14> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < ROW_LEN * K * SIMDNUM; cycle++) {
#pragma HLS pipeline

      ap_uint<SIMD * IN_BIT * Np> write_data;
      ap_uint<3> rowBufferIdx = startRowBufferIdx + kr_c;
      if (rowBufferIdx >= (K + 1)){
        rowBufferIdx -= (K + 1);
      }

      if ((outRowIdx - (K / 2) + kr_c < 0) || (outRowIdx - (K / 2) + kr_c >= IN_H)) {
        write_data = 0;
      } else {
        write_data = row_buffer[ipe_simd_c][rowBufferIdx][mem_offset + ch_ipe_c];
      }
      out.write(write_data);

      //counters
      ipe_simd_c++;
      if(ipe_simd_c == (IN_PE / SIMD)){
        ipe_simd_c = 0;
        ch_ipe_c++;
        if(ch_ipe_c == INPENUM){
          ch_ipe_c = 0;
          kr_c++;
          if(kr_c == K){
            kr_c = 0;
            mem_offset += INPENUM;
            if(mem_offset == INPENUM * ROW_LEN){
              mem_offset = 0;
            }
          }
        }
      }
    }
  }
}

/*
Reshape logic
Input:  IN_PE * IN_BIT ---> Np ---> IN_W / Np = ROW_LEN ---> IN_CH / IN_PE ---> IN_H
Output: SIMD * IN_BIT * Np ---> IN_CH / SIMD ---> K ---> IN_W / Np = ROW_LEN ---> OUT_CH / PE ---> IN_H
IN_PE >= SIMD
row_buffer[IN_PE / SIMD][K + 1][ROW_LEN * (IN_CH / IN_PE)]
*/

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned OUTPENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned SIMD>
void reshape_buffer_INPE_SIMD(stream<ap_uint<IN_PE * IN_BIT> > &in,
                              stream<ap_uint<SIMD * IN_BIT * Np> > &out,
                              const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)

  ap_uint<SIMD * IN_BIT * Np> row_buffer[IN_PE / SIMD][K + 1][ROW_LEN * (IN_CH / IN_PE)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<3> storeBufferIdx = 0;
  ap_uint<3> loadBufferIdx = 1;
  ap_int<10> rowIdx = - (K - 1);

  for (unsigned rep = 0; rep < reps * IN_H + (K - 1); rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row_INPE_SIMD<K, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, Np, ROW_LEN>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    stream_out_rows_INPE_SIMD<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, Np, ROW_LEN, OUTPENUM>(out, row_buffer, (rep < (K - 1)), rowIdx, loadBufferIdx);
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
#endif

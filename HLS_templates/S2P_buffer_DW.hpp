#ifndef __S2P_BUFFER_DW_HPP__
#define __S2P_BUFFER_DW_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"


//-----------------------------------------------------------------------padding and sliding window--------------------------------------------------------------------
template <unsigned K, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned BufferIdx_bw, unsigned n_c_bw,
          unsigned pe_ipe_c_bw, unsigned ch_pe_c_bw, unsigned mem_offset_bw>
void DW_stream_in_row_PE_INPE(
    stream<ap_uint<IN_PE * IN_BIT> > &in,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)],
    bool skip_flag, ap_uint<BufferIdx_bw> rowBufferIdx){
#pragma HLS inline off
  const unsigned PAD_LEN = (K - 1) / 2;

  if (skip_flag)
    return;

  //counters
  ap_uint<n_c_bw> n_c = 0;
  ap_uint<pe_ipe_c_bw> pe_ipe_c = 0;
  ap_uint<ch_pe_c_bw> ch_pe_c = 0;
  ap_uint<mem_offset_bw> mem_offset = 0;

  ap_uint<IN_PE * IN_BIT * Np> reg = 0;
  for (unsigned peIdx = 0; peIdx < OUT_CH / IN_PE; peIdx++){
    for (unsigned w_counter = 0; w_counter < ROW_LEN * Np; w_counter++){
#pragma HLS pipeline
      reg = reg >> (IN_PE * IN_BIT);

      ap_uint<IN_PE * IN_BIT> data;
      if ((w_counter < PAD_LEN) || (w_counter > (PAD_LEN + IN_W - 1))) {
        data = 0;
      } else {
        data = in.read();
      }

      reg(IN_PE * IN_BIT * Np - 1, IN_PE * IN_BIT * (Np - 1)) = data;

      //counters
      n_c++;
      if(n_c == Np){
        n_c = 0;
        row_buffer[pe_ipe_c][rowBufferIdx][mem_offset + ch_pe_c] = reg;    
        mem_offset++;
        if(mem_offset == ROW_LEN){
          mem_offset = 0;
          pe_ipe_c++;
          if(pe_ipe_c == (PE / IN_PE)){
            pe_ipe_c = 0;
            ch_pe_c += ROW_LEN;
            if(ch_pe_c == ROW_LEN * (OUT_CH / PE)){
              ch_pe_c = 0;
            }
          }
        }
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned PENUM,
          unsigned BufferIdx_bw, unsigned rowIdx_bw, unsigned kr_c_bw, unsigned mem_offset_bw>
void DW_stream_out_rows_PE_INPE_S2P(
    stream<ap_uint<PE * IN_BIT * Np> > &out,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)],
    bool skip_flag, ap_int<rowIdx_bw> outRowIdx, ap_uint<BufferIdx_bw> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned PE_BIT = PE * IN_BIT;

  if (skip_flag)
    return;

  //counters
  ap_uint<kr_c_bw> kr_c = 0;
  ap_uint<mem_offset_bw> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < PENUM; peIdx++) {
    for (unsigned w_counter = 0; w_counter < ROW_LEN * K; w_counter++) {
#pragma HLS pipeline

      ap_uint<PE * IN_BIT * Np> write_data;
      ap_uint<PE * IN_BIT> data[Np];
#pragma HLS array_partition variable = data complete
      ap_uint<IN_PE * IN_BIT * Np> buffer_data[PE / IN_PE];
#pragma HLS array_partition variable = buffer_data complete

      ap_uint<BufferIdx_bw> rowBufferIdx = startRowBufferIdx + kr_c;
      if (rowBufferIdx >= (K + 1)){
        rowBufferIdx -= (K + 1);
      }
      for (unsigned i = 0; i < PE / IN_PE; i++) {
        buffer_data[i] = row_buffer[i][rowBufferIdx][mem_offset];
      }

      if ((outRowIdx - (K / 2) + kr_c < 0) || (outRowIdx - (K / 2) + kr_c >= IN_H)) {
        write_data = 0;
      } else {
        for (unsigned i = 0; i < PE / IN_PE; i++) {
          for (unsigned j = 0; j < Np; j++){
            data[j]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[i]((j+1)*IN_PE_BIT - 1, j*IN_PE_BIT);
          }
        }

        for (unsigned j = 0; j < Np; j++){
          write_data((j+1)*PE_BIT - 1, j*PE_BIT) = data[j];
        }
      }
      out.write(write_data);

      //counters
 
      kr_c++;
      if (kr_c == K){
        kr_c = 0;
        mem_offset++;
        if (mem_offset == ROW_LEN * PENUM){
          mem_offset = 0;
        }
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned PENUM,
          unsigned BufferIdx_bw, unsigned rowIdx_bw, unsigned mem_offset_bw>
void DW_stream_out_rows_PE_INPE_FPT(
    stream<ap_uint<K * PE * IN_BIT * Np> > &out,
    ap_uint<IN_PE * IN_BIT * Np> row_buffer[PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)],
    bool skip_flag, ap_int<rowIdx_bw> outRowIdx, ap_uint<BufferIdx_bw> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete
#pragma HLS array_partition variable = row_buffer dim = 2 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned PE_BIT = PE * IN_BIT;

  if (skip_flag)
    return;

  //counters
  ap_uint<mem_offset_bw> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < PENUM; peIdx++) {
    for (unsigned w_counter = 0; w_counter < ROW_LEN; w_counter++) {
#pragma HLS pipeline

      ap_uint<K * PE * IN_BIT * Np> write_data;

      for(unsigned kr = 0; kr < K; kr++){
        ap_uint<PE * IN_BIT> data[Np];
#pragma HLS array_partition variable = data dim = 1 complete
        ap_uint<IN_PE * IN_BIT * Np> buffer_data[PE / IN_PE];
#pragma HLS array_partition variable = buffer_data complete

        ap_uint<BufferIdx_bw> rowBufferIdx = startRowBufferIdx + kr;
        if (rowBufferIdx >= (K + 1)){
          rowBufferIdx -= (K + 1);
        }

        for (unsigned i = 0; i < PE / IN_PE; i++) {
          buffer_data[i] = row_buffer[i][rowBufferIdx][mem_offset];
        }

        if ((outRowIdx - (K / 2) + kr < 0) || (outRowIdx - (K / 2) + kr >= IN_H)) {
          for(unsigned j = 0; j < Np; j++){
            data[j] = 0;
          }
        } else {
          for (unsigned i = 0; i < PE / IN_PE; i++) {
            for (unsigned j = 0; j < Np; j++){
              data[j]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[i]((j+1)*IN_PE_BIT - 1, j*IN_PE_BIT);
            }
          }
        }

        for (unsigned j = 0; j < Np; j++){
          write_data(j * K * PE_BIT + kr * PE_BIT + PE_BIT - 1, j * K * PE_BIT + kr * PE_BIT) = data[j];
        }
      }

      out.write(write_data);

      // counters
      mem_offset++;
      if (mem_offset == ROW_LEN * PENUM){
        mem_offset = 0;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned PENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned PE, unsigned BufferIdx_bw,
          unsigned rowIdx_bw, unsigned n_c_bw, unsigned pe_ipe_c_bw, unsigned ch_pe_c_bw, unsigned mem_offset_bw,
          unsigned kr_c_bw, unsigned mem_offset_bw_2>
void DW_reshape_buffer_PE_INPE_S2P(stream<ap_uint<IN_PE * IN_BIT> > &in,
                                  stream<ap_uint<PE * IN_BIT * Np> > &out,
                                  const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)
  const unsigned K_offset = (K + 1) / 2;

  ap_uint<IN_PE * IN_BIT * Np> row_buffer[PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<BufferIdx_bw> storeBufferIdx = 0;
  ap_uint<BufferIdx_bw> loadBufferIdx = 1;
  ap_int<rowIdx_bw> rowIdx = - K_offset;

  for (unsigned rep = 0; rep < reps * IN_H + K_offset; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    DW_stream_in_row_PE_INPE<K, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, BufferIdx_bw, n_c_bw,
                            pe_ipe_c_bw, ch_pe_c_bw, mem_offset_bw>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    DW_stream_out_rows_PE_INPE_S2P<K, IN_H, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, PENUM, BufferIdx_bw, rowIdx_bw,
                                  kr_c_bw, mem_offset_bw_2>(out, row_buffer, (rep < K_offset), rowIdx, loadBufferIdx);
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

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned PENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned PE, unsigned BufferIdx_bw,
          unsigned rowIdx_bw, unsigned n_c_bw, unsigned pe_ipe_c_bw, unsigned ch_pe_c_bw, unsigned mem_offset_bw,
          unsigned mem_offset_bw_2>
void DW_reshape_buffer_PE_INPE_FPT(stream<ap_uint<IN_PE * IN_BIT> > &in,
                                  stream<ap_uint<K * PE * IN_BIT * Np> > &out,
                                  const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)
  const unsigned K_offset = (K + 1) / 2;

  ap_uint<IN_PE * IN_BIT * Np> row_buffer[PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 2 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<BufferIdx_bw> storeBufferIdx = 0;
  ap_uint<BufferIdx_bw> loadBufferIdx = 1;
  ap_int<rowIdx_bw> rowIdx = - K_offset;

  for (unsigned rep = 0; rep < reps * IN_H + K_offset; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    DW_stream_in_row_PE_INPE<K, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, BufferIdx_bw, n_c_bw,
                            pe_ipe_c_bw, ch_pe_c_bw, mem_offset_bw>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    DW_stream_out_rows_PE_INPE_FPT<K, IN_H, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, PENUM, BufferIdx_bw,rowIdx_bw,
                                  mem_offset_bw_2>(out, row_buffer, (rep < K_offset), rowIdx, loadBufferIdx);
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

template <unsigned K, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned BufferIdx_bw, unsigned n_c_bw,
          unsigned mem_offset_bw>
void DW_stream_in_row_INPE_PE(
    stream<ap_uint<IN_PE * IN_BIT> > &in,
    ap_uint<PE * IN_BIT * Np> row_buffer[IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)],
    bool skip_flag, ap_uint<BufferIdx_bw> rowBufferIdx){
#pragma HLS inline off
  const unsigned PAD_LEN = (K - 1) / 2;

  if (skip_flag)
    return;

  ap_uint<n_c_bw> n_c = 0;
  ap_uint<mem_offset_bw> mem_offset = 0;
  ap_uint<PE * IN_BIT * Np> reg[IN_PE / PE];
#pragma HLS ARRAY_PARTITION variable = reg dim = 1 complete
  for (unsigned peIdx = 0; peIdx < OUT_CH / IN_PE; peIdx++){
    for (unsigned w_counter = 0; w_counter < ROW_LEN * Np; w_counter++){
#pragma HLS pipeline
      for (unsigned i = 0; i < (IN_PE / PE); i++){
        reg[i] = reg[i] >> (PE * IN_BIT);
      }
      
      ap_uint<IN_PE * IN_BIT> data;
      if ((w_counter < PAD_LEN) || (w_counter >= (PAD_LEN + IN_W))) {
        data = 0;
      } else {
        data = in.read();
      }

      for (unsigned i = 0; i < (IN_PE / PE); i++){
        reg[i](PE*IN_BIT*Np - 1, PE*IN_BIT*(Np-1)) = data(PE*IN_BIT*(i+1) - 1, PE*IN_BIT*i);
      }
      
      n_c++;
      if (n_c == Np){
        n_c = 0;

        for (unsigned i = 0; i < (IN_PE / PE); i++){
          row_buffer[i][rowBufferIdx][mem_offset] = reg[i];
        }

        mem_offset++;
        if (mem_offset == ROW_LEN * OUT_CH / IN_PE){
          mem_offset = 0;
        }
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned OUTPENUM,
          unsigned BufferIdx_bw, unsigned rowIdx_bw, unsigned kr_c_bw, unsigned ch_ipe_c_bw, unsigned ipe_pe_c_bw,
          unsigned mem_offset_bw>
void DW_stream_out_rows_INPE_PE_S2P(
    stream<ap_uint<PE * IN_BIT * Np> > &out,
    ap_uint<PE * IN_BIT * Np> row_buffer[IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)],
    bool skip_flag, ap_int<rowIdx_bw> outRowIdx, ap_uint<BufferIdx_bw> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  if (skip_flag)
    return;

  //counters
  ap_uint<kr_c_bw> kr_c = 0;
  ap_uint<ch_ipe_c_bw> ch_ipe_c = 0;
  ap_uint<ipe_pe_c_bw> ipe_pe_c = 0;
  ap_uint<mem_offset_bw> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned w_counter = 0; w_counter < ROW_LEN * K; w_counter++) {
#pragma HLS pipeline

      ap_uint<PE * IN_BIT * Np> write_data;
      ap_uint<BufferIdx_bw> rowBufferIdx = startRowBufferIdx + kr_c;
      if (rowBufferIdx >= (K + 1)){
        rowBufferIdx -= (K + 1);
      }

      if ((outRowIdx - (K / 2) + kr_c < 0) || (outRowIdx - (K / 2) + kr_c >= IN_H)) {
        write_data = 0;
      } else {
        write_data = row_buffer[ipe_pe_c][rowBufferIdx][mem_offset + ch_ipe_c];
      }
      out.write(write_data);

      //counters
      kr_c++;
      if(kr_c == K){
        kr_c = 0;
        mem_offset++;
        if(mem_offset == ROW_LEN){
          mem_offset = 0;
          ipe_pe_c++;
          if(ipe_pe_c == (IN_PE / PE)){
            ipe_pe_c = 0;
            ch_ipe_c += ROW_LEN;
            if(ch_ipe_c == ROW_LEN * (OUT_CH / IN_PE)){
              ch_ipe_c = 0;
            }
          }
        }
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned PE, unsigned Np, unsigned ROW_LEN, unsigned PENUM,
          unsigned BufferIdx_bw, unsigned rowIdx_bw, unsigned ch_ipe_c_bw, unsigned ipe_pe_c_bw, unsigned mem_offset_bw>
void DW_stream_out_rows_INPE_PE_FPT(
    stream<ap_uint<K * PE * IN_BIT * Np> > &out,
    ap_uint<PE * IN_BIT * Np> row_buffer[IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)],
    bool skip_flag, ap_int<rowIdx_bw> outRowIdx, ap_uint<BufferIdx_bw> startRowBufferIdx) {
#pragma HLS inline off
#pragma HLS array_partition variable = row_buffer dim = 1 complete
#pragma HLS array_partition variable = row_buffer dim = 2 complete

  const unsigned PE_BIT = PE * IN_BIT;

  if (skip_flag)
    return;

  //counters
  ap_uint<ch_ipe_c_bw> ch_ipe_c = 0;
  ap_uint<ipe_pe_c_bw> ipe_pe_c = 0;
  ap_uint<mem_offset_bw> mem_offset = 0;

  for (unsigned peIdx = 0; peIdx < PENUM; peIdx++) {
    for (unsigned w_counter = 0; w_counter < ROW_LEN; w_counter++) {
#pragma HLS pipeline

      ap_uint<K * PE * IN_BIT * Np> write_data;

      for(unsigned kr = 0; kr < K; kr++){
        ap_uint<PE * IN_BIT> data[Np];
#pragma HLS array_partition variable = data dim = 1 complete

        ap_uint<BufferIdx_bw> rowBufferIdx = startRowBufferIdx + kr;
        if (rowBufferIdx >= (K + 1)){
          rowBufferIdx -= (K + 1);
        }

        if ((outRowIdx - (K / 2) + kr < 0) || (outRowIdx - (K / 2) + kr >= IN_H)) {
          for(unsigned j = 0; j < Np; j++){
            data[j] = 0;
          }
        } else {
          for(unsigned j = 0; j < Np; j++){
            data[j] = row_buffer[ipe_pe_c][rowBufferIdx][mem_offset + ch_ipe_c](j*PE_BIT + PE_BIT - 1, j*PE_BIT);
          }
        }

        for(unsigned j = 0; j < Np; j++){
          write_data(j * K * PE_BIT + kr * PE_BIT + PE_BIT - 1, j * K * PE_BIT + kr * PE_BIT) = data[j];
        }
      }

      out.write(write_data);

      // counters
      mem_offset++;
      if(mem_offset == ROW_LEN){
        mem_offset = 0;
        ipe_pe_c++;
        if(ipe_pe_c == (IN_PE / PE)){
          ipe_pe_c = 0;
          ch_ipe_c += ROW_LEN;
          if(ch_ipe_c == ROW_LEN * (OUT_CH / IN_PE)){
            ch_ipe_c = 0;
          }
        }
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned PENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned PE, unsigned BufferIdx_bw,
          unsigned rowIdx_bw, unsigned n_c_bw, unsigned mem_offset_bw, unsigned kr_c_bw, unsigned ch_ipe_c_bw,
          unsigned ipe_pe_c_bw, unsigned mem_offset_bw_2>
void DW_reshape_buffer_INPE_PE_S2P(stream<ap_uint<IN_PE * IN_BIT> > &in,
                                  stream<ap_uint<PE * IN_BIT * Np> > &out,
                                  const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)
  const unsigned K_offset = (K + 1) / 2;

  ap_uint<PE * IN_BIT * Np> row_buffer[IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<BufferIdx_bw> storeBufferIdx = 0;
  ap_uint<BufferIdx_bw> loadBufferIdx = 1;
  ap_int<rowIdx_bw> rowIdx = - K_offset;

  for (unsigned rep = 0; rep < reps * IN_H + K_offset; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    DW_stream_in_row_INPE_PE<K, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, BufferIdx_bw, n_c_bw,
                            mem_offset_bw>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    DW_stream_out_rows_INPE_PE_S2P<K, IN_H, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, PENUM, BufferIdx_bw, rowIdx_bw,
                                  kr_c_bw, ch_ipe_c_bw, ipe_pe_c_bw, mem_offset_bw_2>(out, row_buffer, (rep < K_offset), rowIdx, loadBufferIdx);
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

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned OUT_CH, unsigned PENUM,
          unsigned Np, unsigned IN_BIT, unsigned IN_PE, unsigned PE, unsigned BufferIdx_bw,
          unsigned rowIdx_bw, unsigned n_c_bw, unsigned mem_offset_bw, unsigned ch_ipe_c_bw, unsigned ipe_pe_c_bw,
          unsigned mem_offset_bw_2>
void DW_reshape_buffer_INPE_PE_FPT(stream<ap_uint<IN_PE * IN_BIT> > &in,
                                  stream<ap_uint<K * PE * IN_BIT * Np> > &out,
                                  const unsigned reps = 1) {
  const unsigned ROW_LEN = (IN_W + K - 2) / Np + 1;                                       // ceil((IN_W + K - 1)/Np)
  const unsigned K_offset = (K + 1) / 2;

  ap_uint<PE * IN_BIT * Np> row_buffer[IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 2 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<BufferIdx_bw> storeBufferIdx = 0;
  ap_uint<BufferIdx_bw> loadBufferIdx = 1;
  ap_int<rowIdx_bw> rowIdx = - K_offset;

  for (unsigned rep = 0; rep < reps * IN_H + K_offset; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    DW_stream_in_row_INPE_PE<K, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, BufferIdx_bw, n_c_bw,
                            mem_offset_bw>(in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    DW_stream_out_rows_INPE_PE_FPT<K, IN_H, IN_W, OUT_CH, IN_BIT, IN_PE, PE, Np, ROW_LEN, PENUM, BufferIdx_bw, rowIdx_bw,
                            ch_ipe_c_bw, ipe_pe_c_bw, mem_offset_bw_2>(out, row_buffer, (rep < K_offset), rowIdx, loadBufferIdx);
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

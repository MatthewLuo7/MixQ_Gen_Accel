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
void Conv_Comp_Array(ap_int<WPACK_BIT> wpacks[SIMD], ap_uint<IPACK_BIT> ipacks[SIMD], ap_int<W_BIT + IN_BIT + SIMD_BIT + PA_BIT> DSP_PartialRes[Kp + Np - 1]) {
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
          unsigned SIMD_BIT, unsigned adW_BIT>
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

  ap_int<M_BIT> PartialRes[K + Np - 1][PE];
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 1
#pragma HLS ARRAY_PARTITION variable = PartialRes complete dim = 2

  ap_uint<Np * SIMD * IN_BIT> in_data = 0;
  ap_uint<K * SIMD * W_BIT> cur_weights[PE];
#pragma HLS ARRAY_PARTITION variable = cur_weights complete dim = 1
  ap_uint<3> k_counter = 0;
  ap_uint<11> infold_counter = 0;
  for(unsigned h = 0; h < OUT_H * reps; h++){
    for(unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++){
      for(unsigned cycle = 0; cycle < KNUM * INFOLD * ROW_LEN; cycle++){
#pragma HLS pipeline

        if(k_counter == 0){
          in_data = in.read();
          Conv_Pack_ACT<IN_BIT, SIMD, Np, PROD_BIT, IPACK_BIT>(in_data, ipacks);
          for(unsigned p = 0; p < PE; p++){
            cur_weights[p] = weights[p][peIdx * INFOLD + infold_counter];
          }
        }

        for(unsigned p = 0; p < PE; p++){
          ap_uint<Kp * SIMD * W_BIT> in_weights = cur_weights[p](Kp * SIMD * W_BIT - 1, 0);
          cur_weights[p] = cur_weights[p] >> SIMD * W_BIT;
          Conv_Pack_W<W_BIT, SIMD, Kp, PROD_BIT, WPACK_BIT>(in_weights, wpacks[p]);

          ap_int<W_BIT + IN_BIT + SIMD_BIT + PA_BIT> DSP_PartialRes[Kp + Np - 1];
          #pragma HLS ARRAY_PARTITION variable = DSP_PartialRes complete dim = 1

          Conv_Comp_Array<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, PA_BIT, WPACK_BIT, IPACK_BIT>(wpacks[p], ipacks, DSP_PartialRes);
        }
      }
    }
  }

  for (unsigned int h = 0; h < OUT_H * reps; h++) {
    for (unsigned int peIdx = 0; peIdx < OUTPENUM; peIdx++) {
      for (unsigned int w = 0; w < OUT_W + K - 1; w += 2) {
        for (unsigned int infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          // bool m_clear = (w == 0);
          // bool o_clear = (infoldIdx == 0);
          // bool o_out = (infoldIdx == INFOLD - 1 && w != 0);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          conv3x3_1Dopt_2a_pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack);
          for (unsigned p = 0; p < PE; p++) {
            conv3x3_1Dopt_2a_pack_weight_data<W_BIT, SIMD, PROD_BIT, WPACK_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          for (int p = 0; p < PE; p++) {
            #pragma HLS unroll
            ap_int<W_BIT + IN_BIT + SIMD_BIT>     firPartial0;
            ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> firPartial1;
            ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> firPartial2;
            ap_int<W_BIT + IN_BIT + SIMD_BIT>     firPartial3;

            conv3x3_1Dopt_2a_simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, WPACK_BIT>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2, firPartial3);

            if (o_clear) {
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            } else {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }
          }
          ap_int<M_BIT * PE> oData0;
          ap_int<M_BIT * PE> oData1;

          if (o_out) {
            for (int p = 0; p < PE; p++) {
              oData0((p + 1) * M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              oData1((p + 1) * M_BIT - 1, p * M_BIT) = outPartialArr1[p];
            }
            out.write((oData1, oData0));
          }
        }
      }
    }
  }
}


//---------------------------------------------------------------------data packing and DSP result seperating---------------------------------------------

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void conv3x3_1Dopt_2a_pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS array_partition variable = ipack

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void conv3x3_1Dopt_2a_pack_input_data_overlap(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD], ap_uint<2> ipfix[SIMD]) {
#pragma HLS array_partition variable = ipack
#pragma HLS array_partition variable = ipfix

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
    ipfix[i] = (A[i * IN_BIT], B[i * IN_BIT]);
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT, unsigned WPACK_BIT>
void conv3x3_1Dopt_2a_pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0, ap_int<WPACK_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] = (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT, unsigned WPACK_BIT>
void conv3x3_1Dopt_2a_pack_weight_data_overlap(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0, ap_int<WPACK_BIT> wpack[SIMD], ap_uint<2> wpfix[SIMD]) {
#pragma HLS array_partition variable = wpack
#pragma HLS array_partition variable = wpfix

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] = (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
    wpfix[i] = (w0_seg[0], w1_seg[0]);
  }
}


template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE, unsigned SIMD_BIT, unsigned WPACK_BIT>
void conv3x3_1Dopt_2a_simd_MAC(ap_int<WPACK_BIT> wpack[SIMD], ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<W_BIT + IN_BIT + SIMD_BIT> &partial0, ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> &partial1,
              ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> &partial2, ap_int<W_BIT + IN_BIT + SIMD_BIT> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
  
  ap_int<W_BIT + IN_BIT + SIMD_BIT> r0, r3;
  ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> r1, r2;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;

  for (int i = 0; i < SIMD; i += CASCADE) {
#pragma HLS unroll
    ap_int<PROD_BIT * 4> dspres = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
#pragma HLS unroll
      dspres += wpack[i + cs] * ipack[i + cs];
    }

    ap_int<PROD_BIT - 1> p0 = dspres(PROD_BIT - 2, 0);
    ap_int<PROD_BIT>     p1 = dspres(PROD_BIT * 2 - 1, PROD_BIT) + dspres[PROD_BIT - 1];
    ap_int<PROD_BIT>     p2 = dspres(PROD_BIT * 3 - 1, PROD_BIT * 2) + dspres[PROD_BIT * 2 - 1];
    ap_int<PROD_BIT - 1> p3 = dspres(PROD_BIT * 4 - 2, PROD_BIT * 3) + dspres[PROD_BIT * 3 - 1];

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}


template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned SIMD_BIT, unsigned WPACK_BIT>
void conv3x3_1Dopt_2a_simd_MAC_overlap(ap_int<WPACK_BIT> wpack[SIMD],ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_uint<2> wpfix[SIMD], ap_uint<2> ipfix[SIMD],
              ap_int<PROD_BIT + SIMD_BIT> &partial0, ap_int<PROD_BIT + SIMD_BIT + 1> &partial1,
              ap_int<PROD_BIT + SIMD_BIT + 1> &partial2, ap_int<PROD_BIT + SIMD_BIT> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
#pragma HLS ARRAY_PARTITION variable = wpfix complete
#pragma HLS ARRAY_PARTITION variable = ipfix complete

  ap_int<PROD_BIT + SIMD_BIT> r0, r3;
  ap_int<PROD_BIT + SIMD_BIT + 1> r1, r2;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;

  for (int i = 0; i < SIMD; i ++) {
#pragma HLS unroll

    ap_uint<1> fixsig0 = ((wpfix[i][1] & ipfix[i][0])^(wpfix[i][0] & ipfix[i][1]));
    ap_uint<1> fixsig1 = (wpfix[i][1] & ipfix[i][1]);

    ap_int<PROD_BIT * 4> dspres = wpack[i] * ipack[i];

    ap_int<PROD_BIT + 1> rfix1 = (fixsig0, (ap_uint<PROD_BIT - 1>)0, dspres[PROD_BIT - 1]);
    ap_int<PROD_BIT + 1> rfix2 = (fixsig1, (ap_uint<PROD_BIT - 1>)0, (dspres[PROD_BIT * 2] != fixsig0));
    ap_int<PROD_BIT>     rfix3 = ((ap_uint<PROD_BIT - 1>)0, (dspres[PROD_BIT * 3] != fixsig1));

    ap_int<PROD_BIT>     p0 = dspres(PROD_BIT - 1, 0);
    ap_int<PROD_BIT + 1> p1 = dspres(PROD_BIT * 2, PROD_BIT) + rfix1;
    ap_int<PROD_BIT + 1> p2 = dspres(PROD_BIT * 3, PROD_BIT * 2) + rfix2;
    ap_int<PROD_BIT>     p3 = dspres(PROD_BIT * 4 - 1, PROD_BIT * 3) + rfix3;

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}



template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned SIMD_BIT, unsigned WPACK_BIT>
void conv3x3_1Dopt_2a_simd_MAC_overlap_3R(ap_int<WPACK_BIT> wpack[3][SIMD], ap_uint<PROD_BIT + IN_BIT> ipack[3][SIMD],
              ap_uint<2> wpfix[3][SIMD], ap_uint<2> ipfix[3][SIMD],
              ap_int<PROD_BIT + SIMD_BIT> &partial0, ap_int<PROD_BIT + SIMD_BIT + 1> &partial1,
              ap_int<PROD_BIT + SIMD_BIT + 1> &partial2, ap_int<PROD_BIT + SIMD_BIT> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
#pragma HLS ARRAY_PARTITION variable = wpfix complete
#pragma HLS ARRAY_PARTITION variable = ipfix complete

  ap_int<PROD_BIT + SIMD_BIT> r0, r3;
  ap_int<PROD_BIT + SIMD_BIT + 1> r1, r2;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;

  for(unsigned j = 0; j < 3; j++){
#pragma HLS unroll 
    for(unsigned i = 0; i < SIMD; i ++) {
#pragma HLS unroll

      ap_uint<1> fixsig0 = ((wpfix[j][i][1] & ipfix[j][i][0])^(wpfix[j][i][0] & ipfix[j][i][1]));
      ap_uint<1> fixsig1 = (wpfix[j][i][1] & ipfix[j][i][1]);

      ap_int<PROD_BIT * 4> dspres = wpack[j][i] * ipack[j][i];

      ap_int<PROD_BIT + 1> rfix1 = (fixsig0, (ap_uint<PROD_BIT - 1>)0, dspres[PROD_BIT - 1]);
      ap_int<PROD_BIT + 1> rfix2 = (fixsig1, (ap_uint<PROD_BIT - 1>)0, (dspres[PROD_BIT * 2] != fixsig0));
      ap_int<PROD_BIT>     rfix3 = ((ap_uint<PROD_BIT - 1>)0, (dspres[PROD_BIT * 3] != fixsig1));

      ap_int<PROD_BIT>     p0 = dspres(PROD_BIT - 1, 0);
      ap_int<PROD_BIT + 1> p1 = dspres(PROD_BIT * 2, PROD_BIT) + rfix1;
      ap_int<PROD_BIT + 1> p2 = dspres(PROD_BIT * 3, PROD_BIT * 2) + rfix2;
      ap_int<PROD_BIT>     p3 = dspres(PROD_BIT * 4 - 1, PROD_BIT * 3) + rfix3;

      r0 += p0;
      r1 += p1;
      r2 += p2;
      r3 += p3;
    } 
  }
    
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------convolution array-------------------------------------------------------------

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_W,
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT, unsigned GUARD_BIT,
          unsigned M_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE, 
          unsigned SIMD_BIT, unsigned adW_BIT>
void conv3x3_1Dopt_2a_array_cascade(
    stream<ap_uint<SIMD * IN_BIT * 2> > &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    stream<ap_uint<PE * M_BIT * 2> > &out,
    const unsigned reps = 1) {

  //static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  //static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  //static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2 + adW_BIT;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipack[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack complete dim = 1

  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  for (unsigned int h = 0; h < OUT_H * reps; h++) {
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int w = 0; w < OUT_W + K - 1; w += 2) {
        for (unsigned int infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1 && w != 0);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          conv3x3_1Dopt_2a_pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack);
          for (unsigned p = 0; p < PE; p++) {
            conv3x3_1Dopt_2a_pack_weight_data<W_BIT, SIMD, PROD_BIT, WPACK_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          for (int p = 0; p < PE; p++) {
            #pragma HLS unroll
            ap_int<W_BIT + IN_BIT + SIMD_BIT>     firPartial0;
            ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> firPartial1;
            ap_int<W_BIT + IN_BIT + SIMD_BIT + 1> firPartial2;
            ap_int<W_BIT + IN_BIT + SIMD_BIT>     firPartial3;

            conv3x3_1Dopt_2a_simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE, SIMD_BIT, WPACK_BIT>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2, firPartial3);

            if (o_clear) {
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            } else {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }
          }
          ap_int<M_BIT * PE> oData0;
          ap_int<M_BIT * PE> oData1;

          if (o_out) {
            for (int p = 0; p < PE; p++) {
              oData0((p + 1) * M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              oData1((p + 1) * M_BIT - 1, p * M_BIT) = outPartialArr1[p];
            }
            out.write((oData1, oData0));
          }
        }
      }
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

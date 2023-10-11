#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// using namespace hls;
// #include <iostream>
using namespace std;
#include "stream_tools.h"
#include <assert.h>

template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu_fixed(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                                 ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT + INC_BIT + 1> bn_res = in * inc + bias;
  const ap_uint<OUT_BIT> res_max = ~0;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > res_max) {
      res = res_max;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}
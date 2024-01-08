from string import Template
import math


def find_CASCADE(gb, kp, np, simd, kpf, fp=True):
    cascade = 1
    if fp:
        upper = min((2**gb) // min(kp, np), simd*kpf)
    else:
        upper = min((2**gb), simd*kpf)
    for factor in range(1, int(upper) + 1):
        if (simd*kpf)%factor == 0:
            cascade = factor

    return cascade

FP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
    ''')

KP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
const bool CONV_${No}_PatternFlag = true;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE * CONV_${No}_Kp;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
    ''')

rebuffer_SIMD_INPE_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_INPE_SIMD_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_INPE_SIMD_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_SIMD_INPE_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_K * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_INPE_SIMD_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_K * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_INPE_SIMD_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

FP_array_cascade = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
FP_Array_Cascade<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
                 CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * ${KPF}, CONV_${No}_PE, CONV_${No}_Kp,
                 CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
                 CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')

KP_array_cascade = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
KP_Array_Cascade<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
                 CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * ${KPF}, CONV_${No}_PE, CONV_${No}_Kp,
                 CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
                 CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT, CONV_${No}_PatternFlag>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')

red_bw_temp = Template('''//--------------------Conv ${No}: Decrease Bit-width--------------------
stream<ap_uint<CONV_${No}_ACTP * CONV_${No}_M_BIT> > conv_${No}_dec_bw_out("conv_${No}_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT, CONV_${No}_ACTP * CONV_${No}_M_BIT,
                                  CONV_${No}_DEC_BW_NUM>(conv_${No}_array_out, conv_${No}_dec_bw_out, reps);
    ''')

act_trim_temp = Template('''//--------------------Conv ${No}: Activate and Trim--------------------
stream<ap_uint<CONV_${No}_ACTP * CONV_${No}_OUT_BIT> > conv_${No}_act_out("conv_${No}_act_out");
Activation_Trim<CONV_${No}_K, CONV_${No}_IN_W, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_OUT_CH,
CONV_${No}_IN_BIT, CONV_${No}_OUT_BIT, CONV_${No}_W_BIT, CONV_${No}_INC_BIT, CONV_${No}_BIAS_BIT,
CONV_${No}_L_SHIFT, CONV_${No}_OCH_PF, CONV_${No}_ACTP, CONV_${No}_Np, CONV_${No}_M_BIT>(conv_${No}_dec_bw_out, conv_${No}_inc, conv_${No}_bias, conv_${No}_act_out, reps);
    ''')

inc_bw_temp_1 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT> > conv_${No}_conv_out("conv_${No}_conv_out");
#pragma HLS STREAM variable = conv_${No}_conv_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, 2 * CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT,
CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_conv_out, reps);

#ifdef DEBUG
cout << "conv_${No}_conv_out size " << conv_${No}_conv_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OCH_PF,
                                 CONV_${No_lat}_IN_BIT>(conv_${No}_conv_out, output_path+"conv_${No}_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
stream<ap_uint<CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT> > conv_${No}_layer_out("conv_${No}_layer_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = ${POOL_DEPTH}
max_pool2x2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_BIT,
            CONV_${No}_OCH_PF>(conv_${No}_conv_out, conv_${No}_layer_out, reps);
#ifdef DEBUG
cout << "conv_${No}_pool_out size " << conv_${No}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H / 2, CONV_${No}_IN_W / 2,
                           CONV_${No}_IN_CH, CONV_${No}_OCH_PF, CONV_${No}_OUT_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_pool_out.txt", reps);
#endif
''')

inc_bw_temp_2 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
stream<ap_uint<CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT> > conv_${No}_layer_out("conv_${No}_layer_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_layer_out, reps);

#ifdef DEBUG
cout << "conv_${No}_layer_out size " << conv_${No}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OCH_PF,
                                 CONV_${No_lat}_IN_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_conv_out.txt", reps);
#endif
''')


class Gen_Opt_Templates:
    def __init__(self, PType, conv_n, conv_k, conv_ich, conv_och, conv_col, conv_kp, conv_np, conv_gb, conv_simd, conv_pe, conv_kpf, conv_in_pe, max_pool=False):
        self.PType = PType
        self.conv_n = conv_n
        self.conv_k = conv_k
        self.conv_ich = conv_ich
        self.conv_kp = conv_kp
        self.conv_np = conv_np
        self.conv_gb = conv_gb
        self.conv_simd = conv_simd
        self.conv_kpf = conv_kpf
        self.max_pool = max_pool
        self.conv_och = conv_och
        self.conv_col = conv_col
        self.conv_pe = conv_pe
        self.conv_in_pe = conv_in_pe

    def gen_conv_para(self):
        if self.conv_n == 0:
                in_assign = 'conv0_in'
        else:
                in_assign = f'conv_{self.conv_n-1}_layer_out'

        if self.PType == 0:
            return FP_para.substitute(No=str(self.conv_n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv_k * self.conv_k * self.conv_ich))),
                                      SIMD_BIT=str(math.ceil(math.log2(self.conv_kpf * self.conv_simd * min(self.conv_kp, self.conv_np)))),
                                      CASCADE=str(find_CASCADE(self.conv_gb, self.conv_kp, self.conv_np, self.conv_simd, self.conv_kpf, True)))
        elif self.PType == 1:
            return KP_para.substitute(No=str(self.conv_n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv_k * self.conv_k * self.conv_ich))),
                                      SIMD_BIT=str(math.ceil(math.log2(self.conv_kpf * self.conv_simd))),
                                      CASCADE=str(find_CASCADE(self.conv_gb, self.conv_kp, self.conv_np, self.conv_simd, self.conv_kpf, False)))

    def gen_reshape_buffer(self):
        if self.conv_kpf == 1:
            if self.conv_simd >= self.conv_in_pe:
                return rebuffer_SIMD_INPE_S2P.substitute(No=str(self.conv_n))
            else:
                return rebuffer_INPE_SIMD_S2P.substitute(No=str(self.conv_n))
        else:
            if self.conv_simd >= self.conv_in_pe:
                return rebuffer_SIMD_INPE_FPT.substitute(No=str(self.conv_n))
            else:
                return rebuffer_INPE_SIMD_FPT.substitute(No=str(self.conv_n))

    def gen_conv_array(self):
        if self.PType == 0:
            return FP_array_cascade.substitute(No=str(self.conv_n), KPF=str(self.conv_kpf))
        elif self.PType == 1:
            return KP_array_cascade.substitute(No=str(self.conv_n), KPF=str(self.conv_kpf))

    def gen_reduce_bw(self):
        return red_bw_temp.substitute(No=str(self.conv_n))

    def gen_act_trim(self):
        return act_trim_temp.substitute(No=str(self.conv_n))

    def gen_increase_bw(self):
        if self.max_pool:
            return inc_bw_temp_1.substitute(No=str(self.conv_n), No_lat=str(self.conv_n+1), CONV_DEPTH=str(math.ceil(self.conv_col * self.conv_och / self.conv_pe)),
                                            POOL_DEPTH=str(math.ceil(self.conv_col * self.conv_och / (self.conv_pe * 2))))
        else:
            return inc_bw_temp_2.substitute(No=str(self.conv_n), No_lat=str(self.conv_n+1), CONV_DEPTH=str(math.ceil(self.conv_col * self.conv_och / self.conv_pe)))

    def gen_back(self):
        return back_temp.substitute(No=str(self.conv_n))
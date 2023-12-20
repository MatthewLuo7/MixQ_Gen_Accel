from string import Template
import math
import pickle
import numpy as np

name_mapping_FP = {
    'simd': 'SIMD',
    'pe': 'PE',
    'actp': 'ACTP',
    'kpf': 'KPF',
    'icol': 'IN_W',
    'irow': 'IN_H',
    'ich': 'IN_CH',
    'och': 'OUT_CH',
    'abit': 'IN_BIT',
    'wbit': 'W_BIT',
    'obit': 'OUT_BIT',
    'kp': 'Kp',
    'np': 'Np',
    'gb': 'GUARD_BIT',
    'incbit': 'INC_BIT',
    'biasbit': 'BIAS_BIT',
    'max_pool': 'MAX_POOL'
    }

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
                 CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
                 CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
                 CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
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


inc_bw_temp_1_assignout = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
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
max_pool2x2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_BIT,
            CONV_${No}_OCH_PF>(conv_${No}_conv_out, out, reps);
''')

inc_bw_temp_2_assignout = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, out, reps);
''')


class FP_Opt_Templates:
    def __init__(self, conv):
        self.conv = conv

    ################################################ Search ################################################
    def dsp_operations(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (1 * 1)
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1
        OUTPENUM = self.conv.och // 1

        dsp_operations = KNUM * INFOLD * ROW_LEN * OUTPENUM * self.conv.irow

        return dsp_operations

    def get_actp(self, simd, pe, kpf):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (simd * kpf)
        OUT_PF = self.conv.np * pe
        min_actp = OUT_PF // (KNUM * INFOLD)

        valid_flag = False
        best_actp = OUT_PF

        actp_p_max = math.floor(math.log2(pe))
        for actp_p in range(0, actp_p_max + 1):
            actp = 2 ** actp_p
            if actp >= min_actp:
                valid_flag = True
                best_actp = actp
                break

        return valid_flag, best_actp

    def reshape_buffer_constraints(self, inpe, simd, pe, kpf):
        if kpf * simd >= inpe:
            flag = (kpf * simd % inpe == 0)
            flag = flag and (self.conv.ich * self.conv.k % (kpf * simd) == 0)
        else:
            # flag = (inpe % (kpf * simd) == 0)
            # flag = flag and (self.conv.ich * self.conv.k % inpe == 0)
            flag = False

        return flag

    def check_constraints(self, inpe, simd, pe, kpf):
        flag = self.reshape_buffer_constraints(inpe=inpe, simd=simd, pe=pe, kpf=kpf)
        flag = flag and (self.conv.och % pe == 0)

        return flag

    def get_feature(self, simd, pe, actp, kpf):
        features_list = [simd, pe, actp, kpf]
        for idx, (k, v) in enumerate(name_mapping_FP.items()):
            if idx < 4:
                continue
            features_list.append(getattr(self.conv, k))

        features_list.append(simd * pe * kpf + actp)
        float_features = list(map(float, features_list))

        return float_features

    def Load_Model(self):
        predictor_file = 'E:/Projects/DeepBurning_MixQ/MixQ_Gen_Accel/Generator/operators/ConvOpt_FP/predictors/XGB_dsp.pkl'
        with open(predictor_file, 'rb') as file:
            model_dsp = pickle.load(file)
        
        self.model_dsp = model_dsp

    def predict(self, simd, pe, actp, kpf):
        features = self.get_feature(simd, pe, actp, kpf)
        # print(len(features))
        X = np.array(features).reshape(1, -1)
        y_dsp = int(self.model_dsp.predict(X))

        return y_dsp

    ################################################ Processing ################################################
    def weight_reorder(self):
        if self.conv.kpf == 1:
            w = self.conv.w    # [och, ich, kr, kc]
            assert self.conv.och%self.conv.pe == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}"
            assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

            w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
            w = w.reshape(self.conv.och//self.conv.pe, self.conv.pe, self.conv.k, self.conv.k*self.conv.ich//self.conv.simd, self.conv.simd)  # [och/pe, pe, kc, kr*ich/simd, simd]
            w = w.transpose(1, 0, 3, 2, 4) # [pe, och/pe, kr*ich/simd, kc, simd]
            w = w[:, :, :, ::-1, :]
            w = w.reshape(self.conv.pe, -1, self.conv.k*self.conv.simd)  # [pe, och/pe*kr*ich/simd, kc*simd]
            self.conv.w = w

            return f"const ap_uint<{self.conv.k * self.conv.wbit * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="
        else:
            w = self.conv.w    # [och, ich, kr, kc]
            assert self.conv.och%self.conv.pe == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}"
            assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

            w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
            w = w.reshape(self.conv.och//self.conv.pe, self.conv.pe, self.conv.k, self.conv.k, self.conv.ich//self.conv.simd, self.conv.simd)  # [och/pe, pe, kc, kr, ich/simd, simd]
            w = w.transpose(1, 0, 4, 2, 3, 5) # [pe, och/pe, ich/simd, kc, kr, simd]
            w = w[:, :, :, ::-1, :, :]
            w = w.reshape(self.conv.pe, -1, self.conv.k*self.conv.k*self.conv.simd)  # [pe, och/pe*ich/simd, kc*kr*simd]
            self.conv.w = w

            return f"const ap_uint<{self.conv.k * self.conv.wbit * self.conv.k * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    ################################################ Write Weight Tools ################################################
    def hex_str(self, x):
        return ('"' + hex(x) + '"')

    def pack1d_str(self, arr): # x: 1d-array
        x = 0
        for v in arr[::-1]: # [!] reverse simd pack, it is related to hls implemention
            v = int(v) # use python bignumber, not np.int
            assert -1<<self.conv.wbit-1 <= v < 1<<self.conv.wbit-1, f'got v={v} while wbit={self.conv.wbit}'
            x=(x<<self.conv.wbit) + (v&(2**self.conv.wbit-1))
        return self.hex_str(x)

    def print_ndarray_recursion(self, arr, str_func=str, stop=0):
        content = ''
        if not hasattr(arr, '__iter__') or len(arr.shape) == stop:
            content += str_func(arr)
            return content

        ends = '' if (len(arr.shape)==stop+1) else '\n'
        content += '{'
        for i, item in enumerate(arr):
            content += self.print_ndarray_recursion(item, str_func, stop)
            if i!=len(arr)-1:
                content += (','+ends)
        content += (ends+'}')

        return content

    def write_weights(self):
        content = ''
        content += f"// layer: {self.conv.n}, PE: {self.conv.pe}, SIMD: {self.conv.simd}, wbit: {self.conv.wbit}\n"

        # reorder and output weights
        content += self.weight_reorder()
        content += '\n'
        content += self.print_ndarray_recursion(self.conv.w, self.pack1d_str, stop=1)
        content += ';\n'

        # process batchnorm
        if self.conv.inc is not None:
            self.conv.inc = self.conv.inc.reshape(self.conv.och//self.conv.actp, self.conv.actp).T
            content += f"const ap_int<{self.conv.incbit}> conv_{self.conv.n}_inc[{self.conv.actp}][{self.conv.och//self.conv.actp}]=\n"
            content += self.print_ndarray_recursion(self.conv.inc, self.hex_str)
            content += ';\n'
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            self.conv.bias = self.conv.bias.reshape(self.conv.och//self.conv.actp, self.conv.actp).T
            content += f"const ap_int<{self.conv.biasbit}> conv_{self.conv.n}_bias[{self.conv.actp}][{self.conv.och//self.conv.actp}]=\n"
            content += self.print_ndarray_recursion(self.conv.bias, self.hex_str)
            content += ';\n'

        return content

    ################################################ HLS Template ################################################
    def find_CASCADE(self):
        cascade = 1
        upper = min((2**self.conv.gb) // min(self.conv.kp, self.conv.np), self.conv.simd*self.conv.kpf)
        for factor in range(1, int(upper) + 1):
            if (self.conv.simd*self.conv.kpf)%factor == 0:
                cascade = factor
        return cascade

    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'

        return FP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                   SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd * min(self.conv.kp, self.conv.np)))), CASCADE=str(self.find_CASCADE()))

    def gen_reshape_buffer(self):
        if self.conv.kpf == 1:
            if self.conv.simd >= self.conv.in_pe:
                return rebuffer_SIMD_INPE_S2P.substitute(No=str(self.conv.n))
            else:
                return rebuffer_INPE_SIMD_S2P.substitute(No=str(self.conv.n))
        else:
            if self.conv.simd >= self.conv.in_pe:
                return rebuffer_SIMD_INPE_FPT.substitute(No=str(self.conv.n))
            else:
                return rebuffer_INPE_SIMD_FPT.substitute(No=str(self.conv.n))

    def gen_conv_array(self):
        return FP_array_cascade.substitute(No=str(self.conv.n))

    def gen_reduce_bw(self):
        return red_bw_temp.substitute(No=str(self.conv.n))

    def gen_act_trim(self):
        return act_trim_temp.substitute(No=str(self.conv.n))

    def gen_increase_bw(self):
        if self.conv.max_pool:
            return inc_bw_temp_1.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / self.conv.pe)),
                                            POOL_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * 2))))
        else:
            return inc_bw_temp_2.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / self.conv.pe)))

    def gen_increase_bw_for_sampling(self):
        if self.conv.max_pool:
            return inc_bw_temp_1_assignout.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / self.conv.pe)),
                                            POOL_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * 2))))
        else:
            return inc_bw_temp_2_assignout.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / self.conv.pe)))

    def gen_operator(self):
        content = f'''
/********************************************************************************Convolution {self.conv.n}********************************************************************************/

'''
        content += self.gen_conv_para()
        content += f'\n'
        content += self.gen_reshape_buffer()
        content += f'\n'
        content += self.gen_conv_array()
        content += f'\n'
        content += self.gen_reduce_bw()
        content += f'\n'
        content += self.gen_act_trim()
        content += f'\n'
        content += self.gen_increase_bw()
        content += f'\n'

        return content


    def gen_operator_for_sampling(self):
        content = f'''
/********************************************************************************Convolution {self.conv.n}********************************************************************************/

'''
        content += self.gen_conv_para()
        content += f'\n'
        content += self.gen_reshape_buffer()
        content += f'\n'
        content += self.gen_conv_array()
        content += f'\n'
        content += self.gen_reduce_bw()
        content += f'\n'
        content += self.gen_act_trim()
        content += f'\n'
        content += self.gen_increase_bw_for_sampling()
        content += f'\n'

        return content
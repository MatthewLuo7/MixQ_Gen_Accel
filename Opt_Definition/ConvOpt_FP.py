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
const unsigned CONV_${No}_IN_PE = ${IN_PE};
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
const unsigned CONV_${No}_W_Sep = ${W_Sep};
const unsigned CONV_${No}_A_Sep = ${A_Sep};
    ''')

rebuffer_SIMD_INPE_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_SIMD_INPE_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD, ${BufferIdx_bw},
                             ${rowIdx_bw}, ${n_c_bw}, ${simd_ipe_c_bw}, ${ch_simd_c_bw}, ${mem_offset_bw},
                             ${kr_c_bw}, ${simd_c_bw}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_INPE_SIMD_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_INPE_SIMD_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD, ${BufferIdx_bw},
                             ${rowIdx_bw}, ${n_c_bw}, ${mem_offset_bw}, ${kr_c_bw}, ${ch_ipe_c_bw}, ${ipe_simd_c_bw}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_SIMD_INPE_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_K * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_SIMD_INPE_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD, ${BufferIdx_bw},
                             ${rowIdx_bw}, ${n_c_bw}, ${simd_ipe_c_bw}, ${ch_simd_c_bw}, ${mem_offset_bw}, ${simd_c_bw}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

rebuffer_INPE_SIMD_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_K * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_INPE_SIMD_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                             CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD, ${BufferIdx_bw},
                             ${rowIdx_bw}, ${n_c_bw}, ${mem_offset_bw}, ${ch_ipe_c_bw}, ${ipe_simd_c_bw}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

FP_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
FP_Array<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
         CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
         CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
         CONV_${No}_SIMD_BIT, CONV_${No}_W_Sep, CONV_${No}_A_Sep,
         ${k_counter_bw}, ${infold_counter_bw}, ${res_offset_bw}, ${add_offset_bw}>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
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
CONV_${No}_L_SHIFT, CONV_${No}_OCH_PF, CONV_${No}_ACTP, CONV_${No}_Np, CONV_${No}_M_BIT,
${ACTP_NUM_counter_bw}, ${w_counter_bw}, ${add_offset_bw}>(conv_${No}_dec_bw_out, conv_${No}_inc, conv_${No}_bias, conv_${No}_act_out, reps);
    ''')

inc_bw_temp_1 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
stream<ap_uint<2 * CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT> > conv_${No}_conv_out("conv_${No}_conv_out");
#pragma HLS STREAM variable = conv_${No}_conv_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, 2 * CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT,
CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_conv_out, reps);

#ifdef DEBUG
cout << "conv_${No}_conv_out size " << conv_${No}_conv_out.size() << endl;
print_mavu_DSPopt_stream_through_a2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OCH_PF,
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
                           CONV_${No}_OUT_CH, CONV_${No}_OCH_PF, CONV_${No}_OUT_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_pool_out.txt", reps);
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
print_mavu_DSPopt_stream_through_a2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OCH_PF,
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

    def opt_type(self):
        return {'Packing': 'FP', 'DW': False, 'LUT': False}

    def get_opf(self):
        if hasattr(self.conv, 'pe') and self.conv.pe is not None:
            return self.conv.pe
        else:
            return None

    def get_factors(self, m):
        factors = []
        for i in range(1, m + 1):
            if m % i == 0:
                factors.append(i)

        return factors

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

        if pe < min_actp:
            return None

        best_actp = pe
        for actp in self.get_factors(pe):
            if actp >= min_actp:
                best_actp = actp
                break

        return best_actp

    def get_actp_(self):
        C1 = hasattr(self.conv, 'simd') and self.conv.simd is not None
        C2 = hasattr(self.conv, 'pe') and self.conv.pe is not None
        C3 = hasattr(self.conv, 'kpf') and self.conv.kpf is not None

        if C1 and C2 and C3:
            return self.get_actp(self.conv.simd, self.conv.pe, self.conv.kpf)
        else:
            raise TypeError(f'Parallelism factors are not all instantiated!')
            return False

    def reshape_buffer_constraints(self, inpe, simd, kpf):
        C1 = (inpe <= self.conv.ich) and (self.conv.ich % inpe == 0)
        C2 = (kpf == 1) or (kpf == self.conv.k)
        C3 = self.conv.ich % simd == 0
        C4 = (simd % inpe == 0) if (simd >= inpe) else (inpe % simd == 0)

        C5 = (inpe * self.conv.abit) <= 1024
        C6 = (kpf * simd * self.conv.abit * self.conv.np) <= 1024
        C7 = (inpe * self.conv.abit * self.conv.np <= 1024) if (simd >= inpe) else (simd * self.conv.abit * self.conv.np <= 1024)

        flag = C1 and C2 and C3 and C4 and C5 and C6 and C7

        return flag

    def ACT_constraints(self, opf, actp):
        C1 = (opf <= self.conv.och) and (self.conv.och % opf == 0)
        C2 = (actp <= opf) and (opf % actp == 0)
        C3 = actp * self.conv.incbit <= 1024
        C4 = actp * self.conv.biasbit <= 1024

        flag = C1 and C2 and C3 and C4

        return flag

    def opt_constraints(self, inpe, simd, kpf, pe, actp):
        opf = pe
        M_BIT = self.conv.abit + self.conv.wbit + math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))

        C1 = self.reshape_buffer_constraints(inpe, simd, kpf)
        C2 = self.ACT_constraints(opf, actp)

        C3 = (self.conv.k * kpf * simd * self.conv.wbit) <= 1024    # weight width
        C4 = (self.conv.np * opf * M_BIT) <= 1024
        C5 = (2 * opf * self.conv.obit <= 1024) if self.conv.max_pool else (opf * self.conv.obit <= 1024)

        flag = C1 and C2 and C3 and C4 and C5

        return flag

    def opt_constraints_(self):
        C1 = hasattr(self.conv, 'simd') and self.conv.simd is not None
        C2 = hasattr(self.conv, 'pe') and self.conv.pe is not None
        C3 = hasattr(self.conv, 'kpf') and self.conv.kpf is not None
        C4 = hasattr(self.conv, 'actp') and self.conv.actp is not None
        C5 = hasattr(self.conv, 'in_pe') and self.conv.in_pe is not None

        if C1 and C2 and C3 and C4 and C5:
            return self.opt_constraints(inpe=self.conv.in_pe, simd=self.conv.simd, kpf=self.conv.kpf, pe=self.conv.pe, actp=self.conv.actp)
        else:
            raise TypeError(f'Parallelism factors are not all instantiated!')
            return False

    def get_feature(self, simd, pe, actp, kpf):
        attr_names = ['k', 'icol', 'irow', 'ich', 'och', 'max_pool', 'abit', 'wbit', 'obit', 'kp',
                      'np', 'gb', 'w_sep', 'a_sep', 'simd', 'pe', 'actp', 'kpf', 'lshift', 'incbit',
                      'biasbit', 'pack_flag']
        temp_conv = self.conv
        setattr(temp_conv, 'simd', simd)
        setattr(temp_conv, 'pe', pe)
        setattr(temp_conv, 'actp', actp)
        setattr(temp_conv, 'kpf', kpf)

        features_list = []
        for attr in attr_names:
            if hasattr(temp_conv, attr) and (getattr(temp_conv, attr) is not None):
                features_list.append(float(getattr(temp_conv, attr)))

        features_list.append(float(self.conv.cycle))
        features_list.append(float(temp_conv.w_sep * temp_conv.a_sep * simd * pe * kpf + actp))

        return features_list

    def Load_Model(self, predictor_path):
        self.pred_models = {}
        # targets = ['wns', 'dsp', 'lut', 'bram', 'II']
        targets = ['dsp', 'lut', 'bram', 'wns']
        for tar in targets:
            predictor_file = predictor_path / f'BRR_{tar}.pkl'
            with open(predictor_file, 'rb') as file:
                self.pred_models[tar] = pickle.load(file)

    def predict(self, simd, pe, actp, kpf, target):
        features = self.get_feature(simd, pe, actp, kpf)
        X = np.array(features).reshape(1, -1)
        y_np = self.pred_models[target].predict(X)
        y = y_np[0]

        return y

    def predict_batch(self, features):
        targets = ['dsp', 'lut', 'bram', 'wns']
        Y = np.zeros((features.shape[0], len(targets)))
        for idx, target in enumerate(targets):
            Y[:, idx] = self.pred_models[target].predict(features)

        return np.maximum(Y, 0.0)


    ################################################ Processing ################################################
    def weight_reorder(self):
        w = self.conv.w    # [och, ich, kr, kc]
        assert self.conv.och%self.conv.pe == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}"
        assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

        w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
        w = w.reshape(self.conv.och//self.conv.pe, self.conv.pe, self.conv.k, self.conv.k // self.conv.kpf, self.conv.kpf, self.conv.ich//self.conv.simd, self.conv.simd)  # [och/pe, pe, kc, kr/kpf, kpf, ich/simd, simd]
        w = w.transpose(1, 0, 3, 5, 2, 4, 6) # [pe, och/pe, kr/kpf, ich/simd, kc, kpf, simd]
        w = w[:, :, :, :, ::-1, :, :]
        w = w.reshape(self.conv.pe, -1, self.conv.k*self.conv.kpf*self.conv.simd)  # [pe, och/pe*kr/kpf*ich/simd, kc*kpf*simd]
        self.conv.w = w

        return f"const ap_uint<{self.conv.k * self.conv.wbit * self.conv.kpf * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    def weight_shape(self):
        return (self.conv.pe, (self.conv.och // self.conv.pe) * (self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)), self.conv.k * self.conv.kpf * self.conv.simd)

    def ceil_width(self, x, min_BW=1):
        x = x + 1
        BW = math.ceil(math.log2(x))
        BW = max(BW, min_BW)

        return BW

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
        if (2**self.conv.gb) < min(self.conv.kp, self.conv.np):
            return cascade
        upper = min((2**self.conv.gb) // min(self.conv.kp, self.conv.np), self.conv.simd*self.conv.kpf)
        for factor in range(1, int(upper) + 1):
            if (self.conv.simd*self.conv.kpf)%factor == 0:
                cascade = factor
        return cascade

    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
                IN_PE = str(self.conv.in_pe) if hasattr(self.conv, 'in_pe') else '3'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'
                IN_PE = f'CONV_{self.conv.n-1}_OCH_PF'

        return FP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                   SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd * min(self.conv.kp, self.conv.np)))), CASCADE=str(self.find_CASCADE()),
                                   W_Sep=self.conv.w_sep, A_Sep=self.conv.a_sep, IN_PE=IN_PE)

    def gen_reshape_buffer(self):
        BufferIdx_bw = self.ceil_width(self.conv.k + 1)
        rowIdx_bw = self.ceil_width(self.conv.irow - 1) + 1
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        if self.conv.kpf == 1:
            if self.conv.simd >= self.conv.in_pe:

                n_c_bw = self.ceil_width(self.conv.np)
                simd_ipe_c_bw = self.ceil_width(self.conv.simd // self.conv.in_pe)
                ch_simd_c_bw = self.ceil_width(self.conv.ich // self.conv.simd)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.ich // self.conv.simd)
                kr_c_bw = self.ceil_width(self.conv.k)
                simd_c_bw = self.ceil_width(self.conv.ich // self.conv.simd)

                return rebuffer_SIMD_INPE_S2P.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                         n_c_bw=str(n_c_bw), simd_ipe_c_bw=str(simd_ipe_c_bw), ch_simd_c_bw=str(ch_simd_c_bw),
                                                         mem_offset_bw=str(mem_offset_bw), kr_c_bw=str(kr_c_bw), simd_c_bw=str(simd_c_bw))
            else:

                n_c_bw = self.ceil_width(self.conv.np)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.ich // self.conv.in_pe)
                kr_c_bw = self.ceil_width(self.conv.k)
                ch_ipe_c_bw = self.ceil_width(self.conv.ich // self.conv.in_pe)
                ipe_simd_c_bw = self.ceil_width(self.conv.in_pe // self.conv.simd)

                return rebuffer_INPE_SIMD_S2P.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                         n_c_bw=str(n_c_bw), mem_offset_bw=str(mem_offset_bw), kr_c_bw=str(kr_c_bw),
                                                         ch_ipe_c_bw=str(ch_ipe_c_bw), ipe_simd_c_bw=str(ipe_simd_c_bw))
        else:
            if self.conv.simd >= self.conv.in_pe:

                n_c_bw = self.ceil_width(self.conv.np)
                simd_ipe_c_bw = self.ceil_width(self.conv.simd // self.conv.in_pe)
                ch_simd_c_bw = self.ceil_width(self.conv.ich // self.conv.simd)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.ich // self.conv.simd)
                simd_c_bw = self.ceil_width(self.conv.ich // self.conv.simd)

                return rebuffer_SIMD_INPE_FPT.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                         n_c_bw=str(n_c_bw), simd_ipe_c_bw=str(simd_ipe_c_bw), ch_simd_c_bw=str(ch_simd_c_bw),
                                                         mem_offset_bw=str(mem_offset_bw), simd_c_bw=str(simd_c_bw))
            else:

                n_c_bw = self.ceil_width(self.conv.np)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.ich // self.conv.in_pe)
                ch_ipe_c_bw = self.ceil_width(self.conv.ich // self.conv.in_pe)
                ipe_simd_c_bw = self.ceil_width(self.conv.in_pe // self.conv.simd)

                return rebuffer_INPE_SIMD_FPT.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                         n_c_bw=str(n_c_bw), mem_offset_bw=str(mem_offset_bw), ch_ipe_c_bw=str(ch_ipe_c_bw),
                                                         ipe_simd_c_bw=str(ipe_simd_c_bw))

    def S2P_buffer_shape(self):
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        if self.conv.simd >= self.conv.in_pe:
            return (self.conv.simd // self.conv.in_pe, self.conv.k + 1, ROW_LEN * (self.conv.ich // self.conv.simd)) # [SIMD / IN_PE][K + 1][ROW_LEN * (IN_CH / SIMD)]
        else:
            return (self.conv.in_pe // self.conv.simd, self.conv.k + 1, ROW_LEN * (self.conv.ich // self.conv.in_pe)) # [IN_PE / SIMD][K + 1][ROW_LEN * (IN_CH / IN_PE)]

    def gen_conv_array(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)
        OUTPENUM = self.conv.och // self.conv.pe

        k_counter_bw = self.ceil_width(KNUM)
        infold_counter_bw = self.ceil_width(INFOLD, min_BW=2)
        res_offset_bw = self.ceil_width(self.conv.kp * KNUM)
        add_offset_bw = self.ceil_width(OUTPENUM * INFOLD)

        return FP_array.substitute(No=str(self.conv.n), k_counter_bw=str(k_counter_bw), infold_counter_bw=str(infold_counter_bw),
                                  res_offset_bw=str(res_offset_bw), add_offset_bw=str(add_offset_bw))

    def gen_reduce_bw(self):
        return red_bw_temp.substitute(No=str(self.conv.n))

    def gen_act_trim(self):
        OPF = self.conv.pe
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        ACTP_NUM_counter_bw = self.ceil_width(OPF / self.conv.actp)
        w_counter_bw = self.ceil_width(self.conv.np * ROW_LEN)
        add_offset_bw = self.ceil_width(self.conv.och // self.conv.actp)

        return act_trim_temp.substitute(No=str(self.conv.n), ACTP_NUM_counter_bw=str(ACTP_NUM_counter_bw),
                                        w_counter_bw=str(w_counter_bw), add_offset_bw=str(add_offset_bw))

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
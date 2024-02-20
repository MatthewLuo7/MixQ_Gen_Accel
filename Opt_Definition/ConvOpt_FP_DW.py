from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math
import pickle
import numpy as np

name_mapping_FP = {
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

FP_DW_para = Template('''//--------------------Conv ${No}: Parameters--------------------
const unsigned CONV_${No}_IN_PE = ${IN_PE};
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_KPF_BIT = ${KPF_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
const unsigned CONV_${No}_W_Sep = ${W_Sep};
const unsigned CONV_${No}_A_Sep = ${A_Sep};
    ''')

DW_rebuffer_PE_INPE_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_PE * CONV_${No}_Np * CONV_${No}_KPF * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
DW_reshape_buffer_PE_INPE_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                              CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_PE,
                              ${BufferIdx_bw}, ${rowIdx_bw}, ${n_c_bw}, ${pe_ipe_c_bw}, ${ch_pe_c_bw}, ${mem_offset_bw}, ${kr_c_bw}, ${mem_offset_bw_2}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

DW_rebuffer_INPE_PE_S2P = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_PE * CONV_${No}_Np * CONV_${No}_KPF * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
DW_reshape_buffer_INPE_PE_S2P<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                              CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_PE, ${BufferIdx_bw},
                              ${rowIdx_bw}, ${n_c_bw}, ${mem_offset_bw}, ${kr_c_bw}, ${ch_ipe_c_bw}, ${ipe_pe_c_bw}, ${mem_offset_bw_2}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

DW_rebuffer_PE_INPE_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_PE * CONV_${No}_Np * CONV_${No}_KPF * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
DW_reshape_buffer_PE_INPE_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                              CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_PE,
                              ${BufferIdx_bw}, ${rowIdx_bw}, ${n_c_bw}, ${pe_ipe_c_bw}, ${ch_pe_c_bw}, ${mem_offset_bw}, ${mem_offset_bw_2}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

DW_rebuffer_INPE_PE_FPT = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_PE * CONV_${No}_Np * CONV_${No}_KPF * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
DW_reshape_buffer_INPE_PE_FPT<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_CH / CONV_${No}_OCH_PF,
                              CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_PE, ${BufferIdx_bw},
                              ${rowIdx_bw}, ${n_c_bw}, ${mem_offset_bw}, ${ch_ipe_c_bw}, ${ipe_pe_c_bw}, ${mem_offset_bw_2}>(conv_${No}_in, conv_${No}_padding_out, reps);
    ''')

FP_array_DW = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
FP_Array_DW<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_OUT_CH,
            CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
            CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
            CONV_${No}_KPF_BIT, CONV_${No}_W_Sep, CONV_${No}_A_Sep,
            ${k_counter_bw}, ${infold_counter_bw}, ${res_offset_bw}, ${add_offset_bw}>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')


class FP_DW_Opt_Templates(FP_Opt_Templates):
    def __init__(self, conv):
        self.conv = conv

    def opt_type(self):
        return {'Packing': 'FP', 'DW': True, 'LUT': False}

    ################################################ Search ################################################
    def dsp_operations(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k // (1 * 1)
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1
        OUTPENUM = self.conv.och // 1

        dsp_operations = KNUM * INFOLD * ROW_LEN * OUTPENUM * self.conv.irow

        return dsp_operations

    def get_actp(self, simd, pe, kpf):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k // kpf
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
        C1 = hasattr(self.conv, 'pe') and self.conv.pe is not None
        C2 = hasattr(self.conv, 'kpf') and self.conv.kpf is not None

        if C1 and C2:
            return self.get_actp(self.conv.pe, self.conv.kpf)
        else:
            raise TypeError(f'Parallelism factors are not all instantiated!')
            return False

    def reshape_buffer_constraints(self, inpe, pe, kpf):
        C1 = (inpe <= self.conv.och) and (self.conv.och % inpe == 0)
        C2 = (kpf == 1) or (kpf == self.conv.k)
        C3 = (pe <= self.conv.och) and (self.conv.och % pe == 0)
        C4 = (pe % inpe == 0) if (pe >= inpe) else (inpe % pe == 0)

        C5 = (inpe * self.conv.abit) <= 1024
        C6 = (kpf * pe * self.conv.abit * self.conv.np) <= 1024
        C7 = (inpe * self.conv.abit * self.conv.np <= 1024) if (pe >= inpe) else (pe * self.conv.abit * self.conv.np <= 1024)

        flag = C1 and C2 and C3 and C4 and C5 and C6 and C7

        return flag

    def opt_constraints(self, inpe, simd, kpf, pe, actp):
        opf = pe
        M_BIT = self.conv.abit + self.conv.wbit + math.ceil(math.log2(self.conv.k * self.conv.k))

        C1 = self.reshape_buffer_constraints(inpe, pe, kpf)
        C2 = self.ACT_constraints(opf, actp)

        C3 = (self.conv.k * kpf * self.conv.wbit) <= 1024    # weight width
        C4 = (self.conv.np * opf * M_BIT) <= 1024
        C5 = (2 * opf * self.conv.obit <= 1024) if self.conv.max_pool else (opf * self.conv.obit <= 1024)

        flag = C1 and C2 and C3 and C4 and C5

        return flag

    def opt_constraints_(self):
        C1 = hasattr(self.conv, 'pe') and self.conv.pe is not None
        C2 = hasattr(self.conv, 'kpf') and self.conv.kpf is not None
        C3 = hasattr(self.conv, 'actp') and self.conv.actp is not None
        C4 = hasattr(self.conv, 'in_pe') and self.conv.in_pe is not None

        if C1 and C2 and C3 and C4:
            return self.opt_constraints(inpe=self.conv.in_pe, kpf=self.conv.kpf, pe=self.conv.pe, actp=self.conv.actp)
        else:
            raise TypeError(f'Parallelism factors are not all instantiated!')
            return False

    # def reshape_buffer_constraints(self, inpe, pe, kpf):
    #     if pe >= inpe:
    #         flag = (pe % inpe == 0)
    #         flag = flag and (self.conv.och % pe == 0)
    #     else:
    #         flag = (inpe % pe == 0)
    #         flag = flag and (self.conv.och % inpe == 0)

    #     return flag

    # def check_constraints(self, inpe, pe, kpf):
    #     flag = self.reshape_buffer_constraints(inpe=inpe, pe=pe, kpf=kpf)
    #     flag = flag and (self.conv.och % pe == 0)

    #     return flag


    ################################################ Processing ################################################
    def weight_reorder(self):
        w = self.conv.w    # [och, ich, kr, kc], ich = 1
        assert self.conv.och%self.conv.pe == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}"
        assert self.conv.w.shape[1] == 1, f'shape: {self.conv.w.shape}'

        w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
        w = w.reshape(self.conv.och//self.conv.pe, self.conv.pe, self.conv.k, self.conv.k//self.conv.kpf, self.conv.kpf)  # [och/pe, pe, kc, kr*/kpf, kpf]
        w = w.transpose(1, 0, 3, 2, 4) # [pe, och/pe, kr/kpf, kc, kpf]
        w = w[:, :, :, ::-1, :]
        w = w.reshape(self.conv.pe, -1, self.conv.k*self.conv.kpf)  # [pe, och/pe*kr/kpf, kc*kpf]
        self.conv.w = w

        return f"const ap_uint<{self.conv.k * self.conv.wbit * self.conv.kpf}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    def weight_shape(self):
        return (self.conv.pe, (self.conv.och // self.conv.pe) * (self.conv.k // self.conv.kpf), self.conv.k * self.conv.kpf)

    ################################################ Write Weight Tools ################################################
    def write_weights(self):
        content = ''
        content += f"// layer: {self.conv.n}, PE: {self.conv.pe}, KPF: {self.conv.kpf}, wbit: {self.conv.wbit}\n"

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
        upper = min((2**self.conv.gb) // min(self.conv.kp, self.conv.np), self.conv.kpf)
        for factor in range(1, int(upper) + 1):
            if (self.conv.kpf)%factor == 0:
                cascade = factor
        return cascade

    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
                IN_PE = str(self.conv.in_pe) if hasattr(self.conv, 'in_pe') else '3'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'
                IN_PE = f'CONV_{self.conv.n-1}_OCH_PF'

        return FP_DW_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k))),
                                     KPF_BIT=str(math.ceil(math.log2(self.conv.kpf * min(self.conv.kp, self.conv.np)))), CASCADE=str(self.find_CASCADE()),
                                     W_Sep=self.conv.w_sep, A_Sep=self.conv.a_sep, IN_PE=IN_PE)

    def gen_reshape_buffer(self):
        BufferIdx_bw = self.ceil_width(self.conv.k + 1)
        rowIdx_bw = self.ceil_width(self.conv.irow - 1) + 1
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        if self.conv.kpf == 1:
            if self.conv.pe >= self.conv.in_pe:

                n_c_bw = self.ceil_width(self.conv.np)
                pe_ipe_c_bw = self.ceil_width(self.conv.pe // self.conv.in_pe)
                ch_pe_c_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.pe)
                mem_offset_bw = self.ceil_width(ROW_LEN)
                kr_c_bw = self.ceil_width(self.conv.k)
                mem_offset_bw_2 = self.ceil_width(ROW_LEN * self.conv.och // self.conv.pe)

                return DW_rebuffer_PE_INPE_S2P.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                          n_c_bw=str(n_c_bw), pe_ipe_c_bw=str(pe_ipe_c_bw), ch_pe_c_bw=str(ch_pe_c_bw),
                                                          mem_offset_bw=str(mem_offset_bw), kr_c_bw=str(kr_c_bw), mem_offset_bw_2=str(mem_offset_bw_2))
            else:

                n_c_bw = self.ceil_width(self.conv.np)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.in_pe)
                kr_c_bw = self.ceil_width(self.conv.k)
                ch_ipe_c_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.in_pe)
                ipe_pe_c_bw = self.ceil_width(self.conv.in_pe // self.conv.pe)
                mem_offset_bw_2 = self.ceil_width(ROW_LEN)

                return DW_rebuffer_INPE_PE_S2P.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                          n_c_bw=str(n_c_bw), mem_offset_bw=str(mem_offset_bw), kr_c_bw=str(kr_c_bw),
                                                          ch_ipe_c_bw=str(ch_ipe_c_bw), ipe_pe_c_bw=str(ipe_pe_c_bw), mem_offset_bw_2=str(mem_offset_bw_2))
        else:
            if self.conv.pe >= self.conv.in_pe:

                n_c_bw = self.ceil_width(self.conv.np)
                pe_ipe_c_bw = self.ceil_width(self.conv.pe // self.conv.in_pe)
                ch_pe_c_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.pe)
                mem_offset_bw = self.ceil_width(ROW_LEN)
                pe_c_bw = self.ceil_width(self.conv.och // self.conv.pe)
                mem_offset_bw_2 = self.ceil_width(ROW_LEN * self.conv.och // self.conv.pe)

                return DW_rebuffer_PE_INPE_FPT.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                          n_c_bw=str(n_c_bw), pe_ipe_c_bw=str(pe_ipe_c_bw), ch_pe_c_bw=str(ch_pe_c_bw),
                                                          mem_offset_bw=str(mem_offset_bw), pe_c_bw=str(pe_c_bw), mem_offset_bw_2=str(mem_offset_bw_2))
            else:

                n_c_bw = self.ceil_width(self.conv.np)
                mem_offset_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.in_pe)
                ch_ipe_c_bw = self.ceil_width(ROW_LEN * self.conv.och // self.conv.in_pe)
                ipe_pe_c_bw = self.ceil_width(self.conv.in_pe // self.conv.pe)
                mem_offset_bw_2 = self.ceil_width(ROW_LEN)

                return DW_rebuffer_INPE_PE_FPT.substitute(No=str(self.conv.n), BufferIdx_bw=str(BufferIdx_bw), rowIdx_bw=str(rowIdx_bw),
                                                          n_c_bw=str(n_c_bw), mem_offset_bw=str(mem_offset_bw), ch_ipe_c_bw=str(ch_ipe_c_bw),
                                                          ipe_pe_c_bw=str(ipe_pe_c_bw), mem_offset_bw_2=str(mem_offset_bw_2))

    def S2P_buffer_shape(self):
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        if self.conv.pe >= self.conv.in_pe:
            return (self.conv.pe // self.conv.in_pe, self.conv.k + 1, ROW_LEN * (self.conv.och // self.conv.pe)) # [PE / IN_PE][K + 1][ROW_LEN * (OUT_CH / PE)]
        else:
            return (self.conv.in_pe // self.conv.pe, self.conv.k + 1, ROW_LEN * (self.conv.och // self.conv.in_pe)) # [IN_PE / PE][K + 1][ROW_LEN * (OUT_CH / IN_PE)]

    def gen_conv_array(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k // self.conv.kpf
        PENUM = self.conv.och // self.conv.pe

        k_counter_bw = self.ceil_width(KNUM)
        infold_counter_bw = self.ceil_width(INFOLD, min_BW=2)
        res_offset_bw = self.ceil_width(self.conv.kp * KNUM)
        add_offset_bw = self.ceil_width(PENUM * INFOLD)

        return FP_array_DW.substitute(No=str(self.conv.n), k_counter_bw=str(k_counter_bw), infold_counter_bw=str(infold_counter_bw),
                                  res_offset_bw=str(res_offset_bw), add_offset_bw=str(add_offset_bw))

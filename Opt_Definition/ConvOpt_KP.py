from string import Template
from ConvOpt_FP import FP_Opt_Templates, act_trim_temp, inc_bw_temp_1_assignout, inc_bw_temp_2_assignout, inc_bw_temp_1, inc_bw_temp_2
import math
import pickle
import numpy as np

name_mapping_KP = {
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
    'max_pool': 'MAX_POOL',
    'pack_flag': 'PACK_FLAG'
    }

KP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
const unsigned CONV_${No}_IN_PE = ${IN_PE};
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const bool CONV_${No}_PatternFlag = ${PatternFlag};
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE * CONV_${No}_Kp;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
const unsigned CONV_${No}_W_Sep = ${W_Sep};
const unsigned CONV_${No}_A_Sep = ${A_Sep};
    ''')

KP_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
KP_Array<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
         CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
         CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
         CONV_${No}_SIMD_BIT, CONV_${No}_W_Sep, CONV_${No}_A_Sep, CONV_${No}_PatternFlag,
         ${kc_counter_bw}, ${kich_counter_bw}, ${och_offset_bw}>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')


class KP_Opt_Templates(FP_Opt_Templates):
    def opt_type(self):
        return {'Packing': 'KP', 'DW': False, 'LUT': False}

    def get_opf(self):
        if hasattr(self.conv, 'pe') and hasattr(self.conv, 'kp'):
            return self.conv.pe * self.conv.kp
        else:
            return None

    ################################################ Search ################################################
    def dsp_operations(self):
        K = self.conv.k
        INFOLD = self.conv.k * self.conv.ich // (1 * 1)
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1
        OUTPENUM = self.conv.och // (1 * self.conv.kp);

        dsp_operations = K * INFOLD * ROW_LEN * OUTPENUM * self.conv.irow

        return dsp_operations

    def get_actp(self, simd, pe, kpf):
        K = self.conv.k
        INFOLD = self.conv.k * self.conv.ich // (simd * kpf)
        OUT_PF = self.conv.np * pe * self.conv.kp
        min_actp = OUT_PF // (K * INFOLD)

        if self.conv.kp * pe < min_actp:
            return None

        best_actp = pe * self.conv.kp

        for actp in self.get_factors(pe * self.conv.kp):
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

    def opt_constraints(self, inpe, simd, kpf, pe, actp):
        opf = pe * self.conv.kp
        M_BIT = self.conv.abit + self.conv.wbit + math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))

        C1 = self.reshape_buffer_constraints(inpe, simd, kpf)
        C2 = self.ACT_constraints(opf, actp)

        C3 = (kpf * simd * self.conv.kp * self.conv.wbit) <= 1024    # weight width
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


    ################################################ Processing ################################################
    def weight_reorder(self):
        w = self.conv.w    # [och, ich, kr, kc]
        assert self.conv.och%(self.conv.pe*self.conv.kp) == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}, kp {self.conv.kp}"
        assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

        w = w.reshape(self.conv.och//(self.conv.kp*self.conv.pe), self.conv.pe, self.conv.kp, self.conv.ich//self.conv.simd, self.conv.simd, self.conv.k // self.conv.kpf, self.conv.kpf, self.conv.k)   # [och/(kp*pe), pe, kp, ich/simd, simd, kr/kpf, kpf, kc]
        w = w.transpose(1, 0, 5, 3, 7, 6, 4, 2)            # [pe, och/(kp*pe), kr/kpf, ich/simd, kc, kpf, simd, kp]
        w = w[:, :, :, :, ::-1, :, :, :]
        w = w.reshape(self.conv.pe, -1, self.conv.kpf*self.conv.simd*self.conv.kp)   # [pe, och/(kp*pe) * kr/kpf * ich/simd * kc, kpf * simd * kp]
        self.conv.w = w

        return f"const ap_uint<{self.conv.wbit * self.conv.kp * self.conv.kpf * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    def weight_shape(self):
        return (self.conv.pe, (self.conv.och // (self.conv.pe * self.conv.kp)) * (self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)) * self.conv.k, self.conv.kp * self.conv.kpf * self.conv.simd)

    ################################################ HLS Template ################################################
    def find_CASCADE(self):
        cascade = 1
        if self.conv.gb < 0:
            return cascade
        upper = min((2**self.conv.gb), self.conv.simd*self.conv.kpf)
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
            
        if self.conv.pack_flag:
            PatternFlag = 'true'
        else:
            PatternFlag = 'false'
        return KP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                  SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd))),
                                  CASCADE=str(self.find_CASCADE()), PatternFlag=PatternFlag, W_Sep=self.conv.w_sep, A_Sep=self.conv.a_sep, IN_PE=IN_PE)

    def gen_conv_array(self):
        INFOLD = self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)
        OUTPENUM = self.conv.och // (self.conv.pe * self.conv.kp)

        kc_counter_bw = self.ceil_width(self.conv.k)
        kich_counter_bw = self.ceil_width(self.conv.k * INFOLD, min_BW=2)
        och_offset_bw = self.ceil_width(OUTPENUM * self.conv.k * INFOLD)

        return KP_array.substitute(No=str(self.conv.n), kc_counter_bw=str(kc_counter_bw), kich_counter_bw=str(kich_counter_bw), och_offset_bw=str(och_offset_bw))

    def gen_act_trim(self):
        OPF = self.conv.pe * self.conv.kp
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        ACTP_NUM_counter_bw = self.ceil_width(OPF / self.conv.actp)
        w_counter_bw = self.ceil_width(self.conv.np * ROW_LEN)
        add_offset_bw = self.ceil_width(self.conv.och // self.conv.actp)

        return act_trim_temp.substitute(No=str(self.conv.n), ACTP_NUM_counter_bw=str(ACTP_NUM_counter_bw),
                                        w_counter_bw=str(w_counter_bw), add_offset_bw=str(add_offset_bw))

    def gen_increase_bw(self):
        if self.conv.max_pool:
            return inc_bw_temp_1.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * self.conv.kp))),
                                            POOL_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * 2))))
        else:
            return inc_bw_temp_2.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * self.conv.kp))))

    def gen_increase_bw_for_sampling(self):
        if self.conv.max_pool:
            return inc_bw_temp_1_assignout.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * self.conv.kp))),
                                            POOL_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * 2))))
        else:
            return inc_bw_temp_2_assignout.substitute(No=str(self.conv.n), No_lat=str(self.conv.n+1), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / (self.conv.pe * self.conv.kp))))
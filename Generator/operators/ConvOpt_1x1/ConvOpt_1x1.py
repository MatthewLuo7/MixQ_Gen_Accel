from string import Template
from ConvOpt_KP import KP_Opt_Templates
import math

KP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = 32;
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
const bool CONV_${No}_PatternFlag = ${PatternFlag};
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE * CONV_${No}_Kp;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
const unsigned CONV_${No}_W_Sep = ${W_Sep};
const unsigned CONV_${No}_A_Sep = ${A_Sep};
    ''')

bias_trim_temp = Template('''//--------------------Conv ${No}: Bias and Trim--------------------
stream<ap_uint<CONV_${No}_ACTP * CONV_${No}_OUT_BIT> > conv_${No}_act_out("conv_${No}_act_out");
Bias_Trim<CONV_${No}_K, CONV_${No}_IN_W, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_OUT_CH,
CONV_${No}_OUT_BIT, CONV_${No}_BIAS_BIT,CONV_${No}_OCH_PF, CONV_${No}_ACTP, CONV_${No}_Np,
${ACTP_NUM_counter_bw}, ${w_counter_bw}, ${add_offset_bw}>(conv_${No}_dec_bw_out, conv_${No}_bias, conv_${No}_act_out, reps);
    ''')

inc_bw_last_temp = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
stream<ap_uint<CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT> > conv_${No}_layer_out("conv_${No}_layer_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, CONV_${No}_OCH_PF * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_layer_out, reps);

#ifdef DEBUG
cout << "conv_${No}_layer_out size " << conv_${No}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OCH_PF,
                                 CONV_${No}_OUT_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_conv_out.txt", reps);
#endif

//-------------------- Add Last --------------------
AddLast<CONV_${No}_IN_H * CONV_${No}_IN_W * CONV_${No}_OUT_CH / 2>(conv_${No}_layer_out, out, reps);''')


class Conv1x1_Opt_Templates(KP_Opt_Templates):
    def __init__(self, conv):
        self.conv = conv
        self.conv.obit = 32
    ################################################ HLS Template ################################################
    def gen_conv_para(self):
        if self.conv.n == 0:
            in_assign = 'conv0_in'
        else:
            in_assign = f'conv_{self.conv.n-1}_layer_out'
            
        if self.conv.pack_flag:
            PatternFlag = 'true'
        else:
            PatternFlag = 'false'
        return KP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                  SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd))),
                                  CASCADE=str(self.find_CASCADE()), PatternFlag=PatternFlag, W_Sep=self.conv.w_sep, A_Sep=self.conv.a_sep)
    def gen_bias_trim(self):
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1

        ACTP_NUM_counter_bw = self.ceil_width(self.conv.pe / self.conv.actp)
        w_counter_bw = self.ceil_width(self.conv.np * ROW_LEN)
        add_offset_bw = self.ceil_width(self.conv.och // self.conv.actp)
        
        return bias_trim_temp.substitute(No=str(self.conv.n), ACTP_NUM_counter_bw=str(ACTP_NUM_counter_bw),
                                        w_counter_bw=str(w_counter_bw), add_offset_bw=str(add_offset_bw))

    def gen_bw_last(self):
        return inc_bw_last_temp.substitute(No=str(self.conv.n), CONV_DEPTH=str(math.ceil(self.conv.icol * self.conv.och / self.conv.pe)))

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
        content += self.gen_bias_trim()
        content += f'\n'
        content += self.gen_bw_last()

        return content
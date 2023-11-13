from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math


Conv1x1_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
  hls::stream<ap_uint<32 * CONV_${No}_PE> > conv_${No}_layer_out("conv_${No}_conv_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = 256
  conv1x1_DSPopt<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_IN_BIT,
                 CONV_${No}_OUT_CH, CONV_${No}_W_BIT, CONV_${No}_BIAS_BIT, 32,
                 CONV_${No}_SIMD, CONV_${No}_PE, CONV_${No}_IN_PE>(
      ${in_assign_last}, conv_${No}_w, conv_${No}_bias, conv_${No}_layer_out, reps);
    ''')

class Conv1x1_Opt_Templates(FP_Opt_Templates):

    def __init__(self, conv):
        self.conv = conv

    ################################################ Processing ################################################
    def weight_reorder(self):
        w = self.conv.w    # [och, ich, kr, kc]
        g_ich = w.shape[1]
        assert self.conv.och%(self.conv.pe) == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}"
        assert g_ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {g_ich}, simd {self.conv.simd}"
        w = w.reshape(self.conv.och//(self.conv.pe), self.conv.pe, g_ich//self.conv.simd, self.conv.simd) # [och / pe, pe, ich / simd, simd]
        w = w.transpose(1,0,2,3)  #[pe, och / pe, ich / simd, simd]
        w = w.reshape(self.conv.pe, -1, g_ich//self.conv.simd, self.conv.simd) # [pe, och / pe, ich / simd, simd]
        w = w.reshape(self.conv.pe, -1, self.conv.simd)   # hls format [pe, och/pe * ich/simd, simd]
        self.conv.w = w
        print(' ->', w.shape)

        return f"const ap_uint<{self.conv.wbit * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    ################################################ HLS Template ################################################
    def gen_operator(self):
        in_assign = f'conv_{self.conv.n-1}_layer_out'
        return Conv1x1_array.substitute(No=str(self.conv.n), in_assign_last=in_assign)
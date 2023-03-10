
from thop import profile
import torch
from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.separable_srvgg_arch import SeparableSRVGGNetCompact
from realesrgan.archs.separable_rrdb_arch import SeparableRRDBNet
from torchsummary import summary

# model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=24, num_conv=32, upscale=2, act_type='relu')
shape = (3, 1080, 1920)
# model = SeparableRRDBNet(num_in_ch=3, num_out_ch=3, num_feat=16, num_block=3, num_grow_ch=32, scale=2)
model = SeparableSRVGGNetCompact(num_in_ch=3, num_feat=16)
input = torch.randn((1, shape[0], shape[1], shape[2]))
flop, param = profile(model, (input,))
# summary(model, shape, batch_size=1)
print(flop/10**9, param)
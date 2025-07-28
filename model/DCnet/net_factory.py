import sys
sys.path.append('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/model/compare_model/DCnet/')
from unet import *

def net_factory(net_type="unet", in_chns=3, class_num=1, normalization='batchnorm', has_dropout=True):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "mcnet_kd":
        net = MCNet2d_KD(in_chns=in_chns, class_num=class_num)
    else:
        net = None
    return net

if __name__ == "__main__":
    import time
    import torch
    from thop import profile

    start = time.time()
    network = net_factory(in_chns=1, class_num=1)
    x = torch.zeros((1, 1, 128,128))
    print(network(x).shape)


    flops, params = profile(network, inputs=(x,))
    print("Params=", str(params / 1e6) + '{}'.format("M"))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))

    end = time.time()
    print('程序运行时间为：', end - start, '秒')



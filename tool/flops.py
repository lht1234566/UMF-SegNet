from torchscan.crawler import crawl_module
from fvcore.nn import FlopCountAnalysis


def parse_shapes(input):
    if isinstance(input, list) or isinstance(input,tuple):
        out_shapes = [item.shape[1:] for item in input]
    else:
        out_shapes = input.shape[1:]

    return out_shapes

def flop_counter(model,input):
    try:
        module_info = crawl_module(model, parse_shapes(input))
        flops = sum(layer["flops"] for layer in module_info["layers"])
    except Exception as e:
        print(f'\nflops counter came across error: {e} \n')
        try:
            print('try another counter...\n')
            if isinstance(input, list):
                input = tuple(input)
            flops = FlopCountAnalysis(model, input).total()
        except Exception as e:
            print(e)
            raise e
        else:
            flops = flops / 1e9
            print(f'FLOPs : {flops:.5f} G')
            return flops

    else:
        flops = flops / 1e9
        print(f'FLOPs : {flops:.5f} G')
        return flops

def print_network_params(model,model_name):
    num_params = 0
    if isinstance(model,list):
        for m in model:
            for param in m.parameters():
                num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))

    else:
        for param in model.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))



if __name__ == '__main__':
    import time
    import torch
    from thop import profile
    from model.custom_networks.UTNet.utnet import UTNet
    from model.custom_networks.UNETR.unetr import UNETR
    # from model.compare_model.TransUnet.vit_seg_modeling import TransUNet
    # import model.compare_model.TransUnet.vit_seg_configs as configs
    # from model.compare_model.TransBTS
    from model.SR_DM.sr_segnet_ import SwinUNETR
    # from model.SR_DM_NoDMB.net_ import SwinUNETR
    # 程序代码
    start= time.time()
    model = SwinUNETR(img_size=(256, 256), in_channels=1, out_channels=1).cuda()
    # model = TransUNet(config_vit,img_size=128, in_channels=3, dummy=True, num_classes=1).cuda()
    # model = UNETR().cuda()





    input = torch.randn(1,1,256,256).cuda()
    flops, params = profile(model, inputs=(input,))
    print("Params=", str(params / 1e6) + '{}'.format("M"))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))

    end = time.time()
    print('程序运行时间为：', end - start, '秒')


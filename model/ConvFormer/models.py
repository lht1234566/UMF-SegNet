from imp import IMP_HOOK
import sys
sys.path.append('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/model/compare_model/ConvFormer/')
import torch

from SETR import Setr_ConvFormer, Setr, Setr_deepvit, Setr_cait, Setr_refiner


def get_model(modelname="SETR", img_size=256, img_channel=3, classes=1, assist_slice_number=4):
    if modelname == "SETR":
        model = Setr(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_deepvit":
        model = Setr_deepvit(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_cait":
        model = Setr_cait(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_refiner":
        model = Setr_refiner(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_ConvFormer":
        model = Setr_ConvFormer(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
if __name__ == "__main__":
    network = get_model(img_size=256, img_channel=3, classes=1)
    x = torch.zeros((1, 3, 256,256))
    print(network(x).shape)
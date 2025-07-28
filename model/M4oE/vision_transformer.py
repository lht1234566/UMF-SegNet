# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from datetime import datetime
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import sys
sys.path.append('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/model/compare_model/M4oE/')
from swin_transformer_moe_decoder_encoder import SwinTransformerSys

logger = logging.getLogger(__name__)

# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=3, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.swin_unet = SwinTransformerSys(img_size=256,
                                            patch_size=4,
                                            in_chans=3,
                                            num_classes=[1],
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=2,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False,
                                            )
        # print(self.swin_unet.flops()/1e9)

    def forward(self, x, dataset_id=4, predict_head=(4)):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x, dataset_id, predict_head)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        # logging.info("pretrained_path:{}".format(pretrained_path))
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            # filter gating keys and decoder keys
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if 'gating' not in k and 'layers_up' not in k
            }
            model_dict = self.swin_unet.state_dict()
            # full_dict = copy.deepcopy(pretrained_dict)
            # for k, v in pretrained_dict.items():
            #     pass
            #     if "layers." in k:
            #         current_layer_num = 3 - int(k[7:8])
            #         current_k = "layers_up." + str(current_layer_num) + k[8:]
            #         full_dict.update({current_k: v})
            # for k in list(full_dict.keys()):
            #     if k in model_dict:
            #         if full_dict[k].shape != model_dict[k].shape:
            #             print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
            #             del full_dict[k]
            full_dict = {}
            for k, v in filtered_dict.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    full_dict[k] = v
                else:
                    print("Skipped loading parameter:", k)

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

if __name__ == "__main__":
    network = SwinUnet()
    x = torch.zeros((1, 3, 64,64))
    print(network(x).shape)
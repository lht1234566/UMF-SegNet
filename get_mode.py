from model.compare_model.ASSNet.ASSNet import ASSNet
from model.compare_model.MLB.Unetplpl_2D import NestedUNet
from model.compare_model.ConvFormer.models import get_model
from model.compare_model.MSA2Net.msa2net import Msa2Net
from model.compare_model.DCnet.net_factory import net_factory
from model.compare_model.Upcol.vnet_AMC import VNet_AMC
from model.compare_model.VPTTA.ResUnet_TTA import ResUnet
from model.compare_model.M4oE.vision_transformer import SwinUnet
from model.SR_DM.sr_segnet import SwinUNETR as SwinUNETR
from model.mynet import SwinUNETR as SwinUNETR_UMFsegnet
from model.SR_DM_NoDMB.net import SwinUNETR as SwinUNETR_NoDMB
from model.SR_DM_nosr.sr_segnet import SwinUNETR as SwinUNETR_nosr
# 'ConvFormer','DCNet','MLB-Seg','UPCoL',
#          'ASSNet','MSA²Net','M⁴oE','VPTTA','UMF-SegNet'
def net(net_type="DcNet"):
    if net_type == "DcNet":
        net = net_factory(in_chns=3, class_num=1).cuda()
    elif net_type == "MLB":
        net = NestedUNet().cuda()
    elif net_type == "ASSNet":
        net =  ASSNet(img_size=(256,256), in_channels=3, out_channels=1).cuda()
    elif net_type == "MSA2Net":
        net = Msa2Net().cuda()
    elif net_type == "M4oE":
        net = SwinUnet().cuda()
    elif net_type == "VPTTA":
        net =  ResUnet(resnet='resnet34', num_classes=1, pretrained=False).cuda()
    elif net_type == "ConvFormer":
        net = get_model(img_size=256, img_channel=3, classes=1).cuda()
    elif net_type == "Upcol":
        net = VNet_AMC().cuda()
    elif net_type == "SR-DM":
        net = SwinUNETR(img_size=(256, 256),in_channels=3,out_channels=1).cuda()
    elif net_type == "UMFsegnet":
        net = SwinUNETR_UMFsegnet(img_size=(256, 256),in_channels=3,out_channels=1).cuda()
    # elif net_type == "SR_DM_nosr":
    #     net = SwinUNETR_nosr(img_size=(256, 256),in_channels=3,out_channels=1).cuda()
    return net
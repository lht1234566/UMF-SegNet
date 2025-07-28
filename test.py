import argparse
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils1.data_utils import *
import dataset

from loss_functions.dice_loss import *

w_sr = 0.5
w_fa = 0.5
from model.compare_model.ASSNet.ASSNet import ASSNet
from model.compare_model.ConvFormer.models import get_model
from model.compare_model.MLB.Unetplpl_2D import NestedUNet
from model.mynet import SwinUNETR
from model.compare_model.Upcol.vnet_AMC import VNet_AMC
from get_mode import net
# from model.new_network import SwinUNETR
# from monai.networks.nets import SwinUNETR


class Trainer(object):
    def __init__(self, args, img_save_path):
        self.args = args
        self.img_save_path = img_save_path
        # Define Dataloader
        _, self.val_loader = dataset.setup_loaders(args)
        # Define network
        # model =ASSNet(args.m)
        self.img_save_path_gt= '/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output6_23/Colon/gt/'
        # model = ASSNet(img_size=(256,256), in_channels=3, out_channels=1).cuda()
        model = net(args.m)
        self.model = model
        resume=args.resume+args.m+'/model/best.pth'
        if resume is not None:
            print(f'load model from {resume}')
            checkpoint = torch.load(resume)
            # print(checkpoint)
            self.model.load_state_dict(checkpoint,strict=False)
            # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
            #                       strict=True)

    def validation(self, epoch):
        self.model.eval()
        for i, sample in enumerate(self.val_loader):
            image_T2,  target = sample
            # if self.args.cuda:
            image_T2, target = image_T2.cuda(),target.cuda()

            with torch.no_grad():
                # input=torch.cat((image_T1, image_T1C,image_T2, image_fl),1)
                # input_ = torch.cat((image_T1, image_T1C, image_T2, image_fl), 1)
                # input=torch.cat((image_T1, image_T1C),1)
                # input_=torch.cat((image_T2, image_fl),1)
                output = self.model(image_T2)
                # print(output.shape)
                out=torch.sigmoid(output)
                target=torch.sigmoid(target)
                # out[out > 0.75] = 225
                # out[out <= 0.75] = 127
                # out[out <= 0.5] = 63
                # out[out <= 0.25] = 0

                # plt.imsave('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output/T1/'+str(i)+'.png',image_T1.cpu().detach().numpy()[0][0],cmap='gray')
                plt.imsave('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output6_23/Colon/T2/'+str(i)+'.png',image_T2.cpu().detach().numpy()[0][0],cmap='gray')
                # plt.imsave('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output/T1C/'+str(i)+'.png',image_T1C.cpu().detach().numpy()[0][0],cmap='gray')
                # plt.imsave('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output/Fl/'+str(i)+'.png',image_fl.cpu().detach().numpy()[0][0],cmap='gray')

                # print(out)
                plt.imsave(self.img_save_path+str(i)+'.png',out.cpu().detach().numpy()[0][0],cmap='gray')
                # plt.imsave(self.img_save_path_gt + str(i) + '.png', target.cpu().detach().numpy()[0][0],cmap='gray')
                # plt.imshow(out.cpu().detach().numpy()[0][0])
                # plt.show()
                if i==50:
                    break
def main():
    parser = argparse.ArgumentParser(description="mulit_task")
    parser.add_argument('--data_dir', type=str,
                        default='/media/ubuntu/Elements SE/data2025.6.3/Task10_Colon/img/',
                        help='dataset path')
    parser.add_argument('--result_path', type=str,
                        default='/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/result_6.20/Colon/',
                        help='result path')

    parser.add_argument('--op', type=str, default='',
                        help='notes')
    parser.add_argument('--base-size', type=int, default=10,
                        help='base image size')
    parser.add_argument('--img-size', type=int, default=256,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--backbone', type=str, default='swinunt',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='BraTs2021',
                        help='dataset name')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='auto',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/result_6.20/Colon/',
                        help='put the path to resuming file if needed')
    # parser.add_argument('--resume', type=str, default=None,help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    if args.test_batch_size is None:
        args.test_batch_size = 1
    if args.lr is None:
        lrs = {
            'BraTs2021': 0.0001
        }
        args.lr = lrs[args.dataset]#'MSA²Net224''ConvFormer256''DCNet','MLB-Seg','UPCol''ASSNet'
    # model_=['ASSNet','DcNet','M4oE','MLB','MSA2Net','MLB','SR-DM','UMFsegnet','Upcol','VPTTA']
    model_ =['ConvFormer']
    for m in model_:
        args.m=m
        sav_path = '/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/output6_23/Colon/'+m+'/'
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        torch.manual_seed(args.seed)
        trainer = Trainer(args, sav_path)

        for epoch in range(0, 1):
            if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.validation(epoch)
        print(m,'done!')


if __name__ == "__main__":
    main()

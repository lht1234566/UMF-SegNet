import argparse
import os
import time
import copy
import numpy
import numpy as np
import torch
from tqdm import tqdm
from utils1.calculate_weights import calculate_weigths_labels
from utils1.lr_scheduler import LR_Scheduler
from utils1.saver import Saver
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss,DiceLoss,DiceFocalLoss
from utils1.summaries import TensorboardSummary
from utils1.metrics import SalEval,dice_coef,PPV,Sensitivity,iou_score,mean_iou
from utils1.data_utils import *
import dataset
import logging
import metric
from loss_functions.dice_loss import *
w_sr=0.5
w_fa=0.5


from model.compare_model.ASSNet.ASSNet import ASSNet
from model.compare_model.MLB.Unetplpl_2D import NestedUNet
from model.compare_model.ConvFormer.models import get_model
from model.compare_model.DCnet.net_factory import net_factory
from model.compare_model.MLB.Unetplpl_2D import NestedUNet
# from model.compare_model.Upcol.vnet_AMC import VNet_AMC
from model.compare_model.VPTTA.ResUnet_TTA import ResUnet
from model.compare_model.M4oE.vision_transformer import SwinUnet
from model.mynet import SwinUNETR #ufm
# from model.SR_DM.sr_segnet import SwinUNETR
from get_mode import net




from model.new_network import SwinUNETR
# from monai.networks.nets import SwinUNETR

class Trainer(object):
    def __init__(self, args,log_path,model_path):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.log_path=log_path
        self.model_path=model_path
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader  = dataset.setup_loaders(args)


        self.nclass=12
        # Define network

        # model=NestedUNet().cuda()
        # model=VNet_AMC().cuda()
        net_type = "UMFsegnet"
        model=net(net_type)
        # model=get_model().cuda()
        # model = SwinUNETR(img_size=(256,256),in_channels=3,out_channels=1,spatial_dims=2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = DiceLoss(to_onehot_y=False, sigmoid=True,squared_pred=True, smooth_nr=0.0, smooth_dr=1e-5)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        # self.evaluator = SalEval()
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.best_pred = 0.0
        self.best_epoch= 0.0
        self.salEvalTrain = metric.SalEval()
        if args.resume is not None:
            print(f'load model from {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                                  strict=True)
        if args.ft:
            args.start_epoch = 0
    def training(self, epoch,logger):
        diceLog = []
        iouLog = []
        psnrLog = []
        sensitivityLog= []
        ssimLog = []
        ppvLog = []
        train_loss = AverageMeter()
        self.model.train()

        with tqdm(total=int(len(self.train_loader) - len(self.train_loader) % self.args.batch_size)) as t:
            t.set_description('Train epoch:{}/{}'.format(epoch, self.args.epochs))
            for i, sample in enumerate(self.train_loader):
                image_T1,image_T1C,image_T2,image_fl, target= sample

                if self.args.cuda:
                    image_T1,image_T1C,image_T2,image_fl, target =image_T1.cuda(), image_T1C.cuda(), image_T2.cuda(), image_fl.cuda(), target.cuda()

                image_T1, image_T1C, image_T2, image_fl, target = image_T1.float(), image_T1C.float(), image_T2.float(), image_fl.float(), target.float()
                # input=torch.cat((image_T1, image_T1C,image_T2, image_fl),1)
                # input_=torch.cat((image_T1, image_T1C,image_T2, image_fl),1)
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                target[target == 255]=1
                target[target == 127] = 2
                target[target == 63] = 3
                # plt.imshow(target.cpu().detach().numpy()[0][0])
                # plt.show()
                output = self.model(image_T2,image_T1)
                # out=torch.sigmoid(output)
                # # break
                #
                # # out=out*3
                # print(image_T2.shape)
                # print(output.shape)
                # plt.imshow(target.cpu().detach().numpy()[0][0])
                # plt.show()
                # plt.imsave()

                loss = self.criterion(output,target)

                self.optimizer.zero_grad()
                # loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
                train_loss.update(loss.item(), len(image_T2))

                # self.salEvalTrain.add_batch(output.data.cpu().numpy(), target.data.cpu().numpy())
                dice = dice_coef(output, target)
                # self.dice_metric(y_pred=output, y=target)
                # dice = self.dice_metric.aggregate().item()

                ppv = PPV(output, target)
                sensitivity = Sensitivity(output, target)
                iou = iou_score(output, target)
                # break

                diceLog.append(dice)
                iouLog.append(iou)
                sensitivityLog.append(sensitivity)
                ppvLog.append(ppv)

                # F_beta = self.evaluator.get_metric()
                t.set_postfix(loss='{:.6f}'.format(train_loss.avg),
                              Dice_avg='{:.6f}'.format(max(diceLog)),
                              iou_avg='{:.6f}'.format(max(iouLog)),
                              Sensitivity_avg='{:.6f}'.format(max(sensitivityLog)),
                              ppv_avg='{:.6f}'.format(max(ppvLog)),
                              )
                t.update(1)
        # self.dice_metric.reset()
        # F_beta = self.evaluator.get_metric()
        logger.info("===> Train: Epoch {} Complete: Loss: {:.6f},Dice:{:.6f},iou:{:.6f},Sensitivity:{:.6f},ppv:{:.6f},\n"
                    .format(epoch, train_loss.avg,max(diceLog), max(iouLog), max(sensitivityLog),max(ppvLog)))
        # self.writer.add_scalar('train/total_loss_epoch', train_loss.avg, epoch)
        print("===> Train: Epoch {} Complete: Loss: {:.6f},,Dice:{:.6f},iou:{:.6f},Sensitivity:{:.6f},ppv:{:.6f},\n"
                            .format(epoch, train_loss.avg,max(diceLog), max(iouLog), max(sensitivityLog),max(ppvLog)))
        #

    def validation(self, epoch):
        test_loss  = AverageMeter()
        diceLog= []
        iouLog= []
        sensitivityLog= []
        ppvLog= []
        best_weights = copy.deepcopy(self.model.state_dict())
        # best_epoch = 0.0

        self.model.eval()
        with tqdm(total=int(len(self.val_loader) - len(self.val_loader) % self.args.test_batch_size)) as t:
            t.set_description('Test epoch:{}/{}'.format(epoch, self.args.epochs))
            for i, sample in enumerate(self.val_loader):
                image_T1, image_T1C, image_T2, image_fl, target = sample
                if self.args.cuda:
                    image_T1, image_T1C, image_T2, image_fl, target = image_T1.cuda(), image_T1C.cuda(), image_T2.cuda(), image_fl.cuda(), target.cuda()
                image_T1, image_T1C, image_T2, image_fl, target = image_T1.float(), image_T1C.float(), image_T2.float(), image_fl.float(), target.float()
                with torch.no_grad():

                    output= self.model(image_T2,image_T1)
                target[target == 255]=1
                target[target == 127] = 2
                target[target == 63] = 3
                # label = torch.argmax(target, dim=1).float()
                loss = self.criterion(output,target)
                test_loss.update(loss.item(), len(image_T2))
                dice = dice_coef(output, target)
                # self.dice_metric(y_pred=output, y=target)
                # dice = self.dice_metric.aggregate().item()
                ppv = PPV(output, target)
                sensitivity = Sensitivity(output, target)
                iou = iou_score(output, target)

                diceLog.append(dice)
                iouLog.append(iou)
                sensitivityLog.append(sensitivity)
                ppvLog.append(ppv)

                t.set_postfix(loss='{:.6f}'.format(test_loss.avg),
                              Dice_avg='{:.6f}'.format(max(diceLog)),
                              iou_avg='{:.6f}'.format(max(iouLog)),
                              Sensitivity_avg='{:.6f}'.format(max(sensitivityLog)),
                              ppv_avg='{:.6f}'.format(max(ppvLog)))
                t.update(1)
        # self.dice_metric.reset()
        print('Validation:')
        print("Loss: {:.6f},Dice:{:.6f},iou:{:.6f},Sensitivity:{:.6f},ppv:{:.6f}"
              .format(test_loss.avg,max(diceLog), max(iouLog), max(sensitivityLog),max(ppvLog)))

        with open(self.log_path + 'Test_Log.txt', 'a') as f:
            f.write('epoch:{}, total_loss_epoch={:.6f},Dice:{:.6f},iou:{:.6f},Sensitivity:{:.6f},ppv:{:.6f}\n'
                    .format(epoch, test_loss.avg,max(diceLog), max(iouLog), max(sensitivityLog),max(ppvLog)))
        f.close()

        new_pred = max(diceLog)
        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.best_epoch = epoch
            best_weights = copy.deepcopy(self.model.state_dict())
        print('best epoch: {}, dice: {:.6f}'.format(self.best_epoch, self.best_pred))
        torch.save(best_weights, os.path.join(self.model_path, 'best.pth'))

def Avg(l):
    return sum(l) / len(l)
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + 'Train_log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
def main():
    parser = argparse.ArgumentParser(description="mulit_task")
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/Luhaotian/Paper-pre/muti-task/data/BraTS2021/img_data',
                        help='dataset path')
    parser.add_argument('--result_path', type=str, default='/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/resulttemp/',
                        help='result path')

    # parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/modelfusion/data/img_data',help='dataset path')
    # parser.add_argument('--result_path', type=str, default='/root/autodl-tmp/modelfusion/code/myproject/result/',help='result path')

    parser.add_argument('--method', type=str, default='ConvFormer',
                        help='method name')
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
    parser.add_argument('--batch-size', type=int, default=2,
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
    parser.add_argument('--resume', type=str, default=None,
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

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'BraTs2021': 100,
        }
        args.epochs = epoches[args.dataset]

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.lr is None:
        lrs = {
            'BraTs2021': 0.0001
        }
        args.lr = lrs[args.dataset]
    log_path = args.result_path + str(args.method) + '_batch' + str(args.batch_size) + '_lr' + str(
        args.lr) + '_epo' + str(args.epochs) + '_op' + str(args.op) + '/result/'
    model_path = args.result_path + str(args.method) + '_batch' + str(args.batch_size) + '_lr' + str(
        args.lr) + '_epo' + str(args.epochs) + '_op' + str(args.op) + '/model/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args,log_path,model_path)
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    logger = gen_log(log_path)
    logger.info("Learning rate:{:f}, batch_size:{}.\n".format(args.lr, args.batch_size))

    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     # trainer.training(epoch,logger)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
    #
    #         trainer.validation(epoch)
    for epoch in range(1):
        trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
   main()

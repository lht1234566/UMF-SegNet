"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# T1,T1C,T2,Fl
# import cv2
import os
import numpy as np
import torch
from PIL import Image
import Transforms as myTransforms
from torch.utils.data import DataLoader
from utils1.data_utils import *
from scipy.ndimage import zoom
from PIL import Image
import torchvision.transforms as transforms
import random

# class GaussianBlur:
#     """
#     Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
#     Adapted from MoCo:
#     https://github.com/facebookresearch/moco/blob/master/moco/loader.py
#     Note that this implementation does not seem to be exactly the same as
#     described in SimCLR.
#     """
#
#     def __init__(self, sigma=[0.1, 2.0]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x
class Dataset(torch.utils.data.Dataset):
    def __init__(self, im_list_t1,im_list_t1c,im_list_t2,im_list_fl, label_list, transform=None):
        self.im_list_t1 = im_list_t1
        self.im_list_t1c = im_list_t1c
        self.im_list_t2 = im_list_t2
        self.im_list_fl = im_list_fl
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, idx):
        # img_t1 = cv2.imread(self.im_list_t1[idx])

        # img_t1 = img_t1[:, :, ::-1]
        img_t1=Image.open(self.im_list_t1[idx]).convert('RGB')
        img_t2 = Image.open(self.im_list_t2[idx]).convert('RGB')
        img_fl = Image.open(self.im_list_fl[idx]).convert('RGB')
        img_t1c = Image.open(self.im_list_t1c[idx]).convert('RGB')
        mask = Image.open(self.label_list[idx]).convert('L')
        mask2 = Image.open(self.label_list[idx]).convert('L')
        mask3 = Image.open(self.label_list[idx]).convert('1')
        mask4 = Image.open(self.label_list[idx]).convert('L')
        # Image Transformations
        if self.transform:
            [img_t1, mask]= self.transform(img_t1,mask)
            [img_t1c, mask2] = self.transform(img_t1c, mask2)
            [img_t2, mask3] = self.transform(img_t2, mask3)
            [img_fl, mask4] = self.transform(img_fl, mask4)
            # print(mask3.shape)
        # plt.imshow(img_t2.cpu().detach().numpy().transpose(1, 2, 0))
        # plt.show()
        # print(img_t2.dtype)
        return img_t1,img_t1c,img_t2 ,img_fl,mask3#np.expand_dims(temp, 0)

    def __len__(self):
        return len(self.im_list_t1)



class LoadData(torch.utils.data.Dataset):
    def __init__(self, data_dir, classes=2, normVal=1.10):
        self.data_dir = data_dir

        self.trainImT1List = list()
        self.trainImT1CList = list()
        self.trainImT2List = list()
        self.trainImflList = list()
        self.trainAnnotList = list()

        self.valImT1List = list()
        self.valImT1CList = list()
        self.valImT2List = list()
        self.valImflList = list()
        self.valAnnotList = list()

    def read_file(self, file_name, train=False):
        if train == True:
            with open(self.data_dir + '/Train/' + file_name, 'r') as textFile:
                train_path=self.data_dir + '/Train/'
                for line in textFile:
                    img_file_T1 = ((train_path).strip() + 'T1/' + line.strip()).strip()
                    img_file_T1C = ((train_path).strip() + 'T1C/' + line.strip()).strip()
                    img_file_T2 = ((train_path).strip() + 'T2/' + line.strip()).strip()
                    img_file_fl = ((train_path).strip() + 'Flair/' + line.strip()).strip()
                    label_file = ((train_path).strip() + 'label/' + line.strip()).strip()
                    # label_img = cv2.imread(label_file, 0) / 255
                    self.trainImT1List.append(img_file_T1)
                    self.trainImT1CList.append(img_file_T1C)
                    self.trainImT2List.append(img_file_T2)
                    self.trainImflList.append(img_file_fl)
                    self.trainAnnotList.append(label_file)
        else:
            with open(self.data_dir + '/Test/' + file_name, 'r') as textFile:
                test_path=self.data_dir + '/Test/'
                for line in textFile:
                    img_file_T1 = ((test_path).strip() + 'T1/' + line.strip()).strip()
                    img_file_T1C = ((test_path).strip() + 'T1C/' + line.strip()).strip()
                    img_file_T2 = ((test_path).strip() + 'T2/' + line.strip()).strip()
                    img_file_fl = ((test_path).strip() + 'Flair/' + line.strip()).strip()
                    label_file = ((test_path).strip() + 'label/' + line.strip()).strip()
                    # label_img = cv2.imread(label_file, 0) / 255
                    self.valImT1List.append(img_file_T1)
                    self.valImT1CList.append(img_file_T1C)
                    self.valImT2List.append(img_file_T2)
                    self.valImflList.append(img_file_fl)
                    self.valAnnotList.append(label_file)
        return 0

    def process_data(self):
        print('Processing training data')
        return_train = self.read_file('data_train.txt', True)

        print('Processing validation data')
        return_val = self.read_file('data_test.txt', False)

        if return_train == 0 and return_val == 0:
            data_dict = dict()
            data_dict['trainImT1'] =self.trainImT1List
            data_dict['trainImT1C'] =self.trainImT1CList
            data_dict['trainImT2'] =self.trainImT2List
            data_dict['trainImfl'] =self.trainImflList
            data_dict['trainAnnot'] = self.trainAnnotList

            data_dict['valImT1'] =self.valImT1List
            data_dict['valImT1C'] =self.valImT1CList
            data_dict['valImT2'] =self.valImT2List
            data_dict['valImfl'] =self.valImflList
            data_dict['valAnnot'] = self.valAnnotList
            return data_dict
        return None


def setup_loaders(args):
    data_loader = LoadData(args.data_dir)
    data = data_loader.process_data()
    if data is None:
        raise ValueError('Error while pickling data. Please check.')

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)#RGB
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)#RGB

    train_transform = myTransforms.Compose([
        myTransforms.Scale(args.img_size, args.img_size),
        myTransforms.RandomCropResize(int(7./224.*args.img_size)),
        myTransforms.RandomFlip(0.5),
        myTransforms.ToTensor(),
        myTransforms.Normalize(mean=mean, std=std),
    ])

    val_transform = myTransforms.Compose([
        myTransforms.Scale(args.img_size, args.img_size),
        myTransforms.ToTensor(),
        myTransforms.Normalize(mean=mean, std=std),
    ])

    # 创建一个变换的组合
    # train_transform = transforms.Compose([
    #     transforms.Scale((args.img_size, args.img_size)),
    #     transforms.Normalize(mean,std),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5),
    #     transforms.RandomGrayscale(p=0.2),
    #     # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    #     transforms.ToTensor()
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Scale((args.img_size, args.img_size)),
    #     transforms.Normalize(mean,std),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5),
    #     transforms.RandomGrayscale(p=0.2),
    #     # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    #     transforms.ToTensor()
    # ])

    train_set = Dataset(data['trainImT1'], data['trainImT1C'],data['trainImT2'],data['trainImfl'],data['trainAnnot'], transform=train_transform)
    val_set = Dataset(data['valImT1'], data['valImT1C'],data['valImT2'],data['valImfl'], data['valAnnot'], transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, drop_last=False)

    return train_loader, val_loader
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/Luhaotian/Paper-pre/muti-task/data/BraTS2021/img_data',
                        help='dataset path')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('--workers', type=int, default=4, help='number of workers to load data')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=240, help='input patch size of network input')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='base image size')
    args = parser.parse_args()
    train_loader, val_loader = setup_loaders(args)
    for i in train_loader:
        image_T1,image_T1C,image_T2,image_fl, target=i
        plt.imshow(image_T2.cpu().detach().numpy()[0].transpose(1, 2, 0))
        plt.show()
        # print(i)
# """
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# """
#
# import cv2
# import os
# import numpy as np
# import torch
# from PIL import Image
# import Transforms as myTransforms
# from torch.utils.data import DataLoader
#
#
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, im_list, label_list, transform=None):
#         self.im_list = im_list
#         self.label_list = label_list
#         self.transform = transform
#
#     def __getitem__(self, idx):
#         img = cv2.imread(self.im_list[idx])
#         img = img[:, :, ::-1]
#         mask = cv2.imread(self.label_list[idx], 0)
#         # print(mask)
#         # Image Transformations
#         if self.transform:
#             [img, mask] = self.transform(img, mask)
#         return (img, mask)
#
#     def __len__(self):
#         return len(self.im_list)
#
#
# class LoadData(torch.utils.data.Dataset):
#     def __init__(self, data_dir, classes=2, normVal=1.10):
#         self.data_dir = data_dir
#
#         self.trainImList = list()
#         self.valImList = list()
#         self.trainAnnotList = list()
#         self.valAnnotList = list()
#
#     def read_file(self, file_name, train=False):
#         with open(self.data_dir + '/' + file_name, 'r') as textFile:
#             for line in textFile:
#                 img_file = ((self.data_dir).strip() + 't2/' + line.strip()).strip()
#                 label_file = ((self.data_dir).strip() + 'label/' + line.strip()).strip()
#                 # label_img = cv2.imread(label_file, 0) / 255
#
#                 if train == True:
#                     self.trainImList.append(img_file)
#                     self.trainAnnotList.append(label_file)
#                 else:
#                     self.valImList.append(img_file)
#                     self.valAnnotList.append(label_file)
#         return 0
#
#     def process_data(self):
#         print('Processing training data')
#         return_train = self.read_file('datatr.txt', True)
#
#         print('Processing validation data')
#         return_val = self.read_file('datate.txt', False)
#
#         if return_train == 0 and return_val == 0:
#             data_dict = dict()
#             data_dict['trainIm'] = self.trainImList
#             data_dict['trainAnnot'] = self.trainAnnotList
#             data_dict['valIm'] = self.valImList
#             data_dict['valAnnot'] = self.valAnnotList
#             return data_dict
#         return None
#
#
# def setup_loaders(args):
#     data_loader = LoadData(args.data_dir)
#     data = data_loader.process_data()
#     if data is None:
#         raise ValueError('Error while pickling data. Please check.')
#
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)#RGB
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)#RGB
#
#     train_transform = myTransforms.Compose([ myTransforms.ToTensor(),
#         myTransforms.Normalize(mean=mean, std=std),
#         myTransforms.Scale(args.img_size, args.img_size),
#         # myTransforms.RandomCropResize(int(7./224.*args.img_size)),
#         # myTransforms.RandomFlip(0.5),
#
#     ])
#
#     val_transform = myTransforms.Compose([myTransforms.ToTensor(),
#         myTransforms.Normalize(mean=mean, std=std),
#         myTransforms.Scale(args.img_size, args.img_size),
#
#     ])
#
#
#     train_set = Dataset(data['trainIm'], data['trainAnnot'], transform=train_transform)
#     val_set = Dataset(data['valIm'], data['valAnnot'], transform=val_transform)
#
#     train_loader = DataLoader(train_set, batch_size=args.batch_size,
#         num_workers=args.workers, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size,
#         num_workers=args.workers, shuffle=False, drop_last=False)
#
#     return train_loader, val_loader#, data['classWeights']
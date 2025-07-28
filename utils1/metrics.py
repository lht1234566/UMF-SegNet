import numpy as np
import torch
import matplotlib.pyplot as plt
# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class SalEval(object):
    def __init__(self, nthresh=99):
        self.nthresh = nthresh
        self.thresh = np.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh)
        self.EPSILON = np.finfo(float).eps

        self.recall = np.zeros((nthresh,))
        self.precision = np.zeros((nthresh,))
        self.mae = 0
        self.num = 0

    def add_batch(self, predict, gth):
        # assert len(predict.shape) == 3 and len(gth.shape) == 3
        for t in range(self.nthresh):
            bi_res = predict > self.thresh[t]
            intersection = np.sum(np.sum(np.logical_and(gth == bi_res, gth), axis=1), axis=1)
            self.recall[t] += np.sum(intersection * 1. / (np.sum(np.sum(gth, axis=1), axis=1) + np.finfo(np.float).eps))
            self.precision[t] += np.sum(intersection * 1. / (np.sum(np.sum(bi_res, axis=1), axis=1) + np.finfo(np.float).eps))

        self.num += gth.shape[0]
    def get_metric(self):
        tr = self.recall / self.num
        tp = self.precision / self.num
        F_beta = (1 + 0.3) * tp * tr / (0.3 * tp + tr + np.finfo(float).eps)
        # jaccard = intersection_sum / (union_sum + smooth)
        # dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

        return np.max(F_beta)


def dice_coef(predict, gth):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict)
    if torch.is_tensor(gth):
        gth = gth.data.cpu().numpy()
    # print(predict)
    # print([[predict>0.25] or [predict<0.5]])
    predict[predict<=0.5] = 0
    # predict[[predict>0.25] and [predict<0.5]] = 0.25
    # predict[[predict>0.5] or [predict<0.75]] = 0.5
    predict[predict > 0.5] = 1

    predict=predict.data.cpu().numpy()
    # print(predict,predict.shape)
    # print(gth,gth.shape)
    # plt.imshow(predict[0][0])
    # plt.show()
    # plt.imshow(gth[0][0])
    # plt.show()
    intersection = (predict * gth).sum()
    # print((2. * intersection + smooth) / \
    #     (predict.sum() + gth.sum() + smooth))
    return (2. * intersection + smooth) / \
        (predict.sum() + gth.sum() + smooth)
# def dice_coef(pred, label):
#     sumdice = 0
#     smooth = 1e-6
#     for i in range(0, 4):
#         pred_bin = (pred == i) * 1
#         label_bin = (label == i) * 1
#         pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
#         label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)
#         intersection = (pred_bin * label_bin).sum()
#         dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
#         sumdice += dice
#     return sumdice / 3


def iou_score(predict, gth):
    smooth = 1e-5

    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(gth):
        gth = gth.data.cpu().numpy()
    output_ = predict > 0.5
    target_ = gth > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
def mean_iou(input, target, classes = 4):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    if torch.is_tensor(input):
        input = input.data.cpu().numpy()
        # print(input)
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    miou = 0
    for i in range(0,classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return  miou/classes




def compute_miou(pred, target, num_classes):
    """
    计算Mean Intersection over Union（mIoU）

    参数：
    - pred: 预测的标签，大小为[N, H, W]，N为批次大小，H为图像高度，W为图像宽度
    - target: 真实标签，大小为[N, H, W]
    - num_classes: 类别数

    返回：
    - miou: mIoU值
    """

    # 将预测和目标标签转换为一维数组
    pred = pred.view(-1)
    target = target.view(-1)

    # 创建混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # 计算预测和目标标签之间的交集和并集
    for p, t in zip(pred, target):
        confusion_matrix[p, t] += 1

    # 计算每个类别的IoU
    intersection = confusion_matrix.diag()
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection

    # 计算每个类别的IoU并求平均
    iou = intersection / union
    miou = iou.mean()

    return miou


def Sensitivity(predict, gth):
    smooth = 1e-5

    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(gth):
        gth = gth.data.cpu().numpy()

    intersection = (predict * gth).sum()

    return (intersection + smooth) / \
        (gth.sum() + smooth)

def PPV(predict, gth):
    smooth = 1e-5

    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(gth):
        gth = gth.data.cpu().numpy()

    intersection = (predict * gth).sum()

    return (intersection + smooth) / \
        (predict.sum() + smooth)
# import numpy as np
#
#
# class Evaluator(object):
#     def __init__(self, num_class):
#         self.num_class = num_class
#         self.confusion_matrix = np.zeros((self.num_class,)*2)
#
#     def Pixel_Accuracy(self):
#         Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
#         return Acc
#
#     def Pixel_Accuracy_Class(self):
#         Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
#         Acc = np.nanmean(Acc)
#         return Acc
#
#     def Mean_Intersection_over_Union(self):
#         MIoU = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix))
#         MIoU = np.nanmean(MIoU)
#         return MIoU
#
#     def Frequency_Weighted_Intersection_over_Union(self):
#         freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
#         iu = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix))
#
#         FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
#         return FWIoU
#
#     def _generate_matrix(self, gt_image, pre_image):
#         mask = (gt_image >= 0) & (gt_image < self.num_class)
#         label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#         count = np.bincount(label, minlength=self.num_class**2,)
#         confusion_matrix = count.reshape(self.num_class, self.num_class)
#         return confusion_matrix
#
#     def add_batch(self, gt_image, pre_image):
#         # print(gt_image.shape,pre_image.shape)
#         assert gt_image.shape == pre_image.shape
#         self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
#
#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_class,) * 2)
#
#
#
#

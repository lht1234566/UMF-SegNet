import numpy as np
import torch
import random
import cv2
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
class Scale(object):
    """
    Resize the given image to a fixed scale
    """
    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # print(np.array(img).shape)
        img = F.resize(img, [self.w, self.h])
        label = F.resize(label, [self.w, self.h], interpolation=Image.NEAREST)
        return [img, label]
class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            img=np.array(img)
            label = np.array(label)
            h, w = img.shape[:2]
            # print(h)
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]
            img_crop=Image.fromarray(img_crop)
            label_crop = Image.fromarray(label_crop)
            img_crop = F.resize(img_crop, [w, h])
            label_crop = F.resize(label_crop, [w, h], interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop
        else:
            # img=Image.fromarray(img)
            # label = Image.fromarray(label)

            return [img, label]

class RandomFlip(object):
    # Randomly flip the given Image with a probability of 0.5
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image.copy())
        if target is not None:
            target = torch.as_tensor(np.expand_dims(np.array(target), 0), dtype=torch.int64)
        return image, target

class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

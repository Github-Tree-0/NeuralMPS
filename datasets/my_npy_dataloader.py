from __future__ import division
import os
import numpy as np
import scipy.io as sio
from imageio import imread
from skimage.transform import resize
import random

import torch
import torch.utils.data as data

from datasets import pms_transforms

class my_npy_dataloader(data.Dataset):
    def __init__(self, args, dir, split='train'):
        if split != 'test':
            self.root   = os.path.join(dir+'_'+split)
        else:
            self.root = os.path.join(dir)
        self.split  = split
        self.args   = args
        objs = np.loadtxt(os.path.join(self.root, 'objects.txt'), dtype=np.str)
        self.objs   = objs
        
        args.log.printWrite('[%s Data] \t%d objs. Root: %s' % (split, len(self.objs), self.root))

    def _getMask(self, obj):
        mask = np.load(os.path.join(self.root, obj, 'mask.npy'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask

    def __getitem__(self, index):
        np.random.seed(index)
        obj   = self.objs[index]
        dirs = np.loadtxt(os.path.join(self.root, obj, 'light_directions.txt'))
        ints = np.loadtxt(os.path.join(self.root, obj, 'light_intensity.txt'))
        img = np.transpose(np.load(os.path.join(self.root, obj, 'imgs.npy')), (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        img_index = np.random.choice(img.shape[2], self.args.input_num, replace=False)
        dirs = dirs[img_index]
        ints = ints[img_index]
        img = img[...,img_index]

        img_shape = img.shape

        if self.args.color_aug:
            img = img * np.random.uniform(1, self.args.color_ratio)

        if self.args.random_ints:
            random_ratio = np.random.uniform(0.1, 1, ints.shape)
            ints = ints * random_ratio # Add random intensity factor
            img = img * random_ratio # Add random intensity factor

        neg_ratio = np.sum(ints<0) / len(ints)
        if neg_ratio > 0.9:
            ints = -ints
        ints = np.repeat(ints, 3)
            
        img = np.tile(img[...,np.newaxis], 3).reshape(img_shape[0], img_shape[1], -1)
        h, w, c = img.shape

        normal_path = os.path.join(self.root, obj, 'normal.npy')
        normal = np.load(normal_path) * 2 - 1 # 这里读入的是非负的normal map

        mask = self._getMask(obj) / 255.0
        img  = img * mask.repeat(img.shape[2], 2)

        if self.args.rescale:
            size = [128, 128]
            img = resize(img, size, order=1, mode='reflect')
            normal = resize(normal, size, order=1, mode='reflect')
            mask = resize(mask, size, order=1, mode='reflect')

        if self.args.crop:
            shape = img.shape
            x1 = random.randint(0, shape[1] - crop_w)
            y1 = random.randint(0, shape[0] - crop_h)
            img = img[y1:y1+crop_h, x1:x1+crop_w]
            normal = normal[y1:y1+crop_h, x1:x1+crop_w]
            mask = mask[y1:y1+crop_h, x1:x1+crop_w]

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4 
        for k in item.keys():
            item[k] = pms_transforms.imgSizeToFactorOfK(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        item['obj'] = obj
        if self.args.random_ints:
            item['random_ints'] = torch.from_numpy(np.repeat(random_ratio, 3)).view(-1, 1, 1).float()
        return item

    def __len__(self):
        return len(self.objs)

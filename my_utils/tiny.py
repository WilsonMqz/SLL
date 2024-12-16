import copy
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage, Resize

from augment.autoaugment_extra import ImageNetPolicy
from augment.cutout import Cutout


class TrainTinyImageNetDataset(Dataset):

    def __init__(self, id, rate_partial=0.3, seed=2):

        np.random.seed(seed)
        random.seed(seed)

        self.id_dict = id
        self.filenames = glob.glob("D:\code\python\weakly_supervised\MCTF_SL/data/tiny-imagenet-200/train/*/*/*.JPEG")
        self.labels = self.get_labels()
        self.seed = seed

        self.transform = Compose([
            Resize(32),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform1 = Compose([
            Resize(32),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            ImageNetPolicy(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.rate_partial = rate_partial
        self.TL, self.SL = self.generate_TL_SL()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path, label, TL, SL = self.filenames[idx], self.labels[idx], self.TL[idx], self.SL[idx]
        img = read_image(img_path)
        if img.shape[0] == 1:
            img = read_image(img_path, ImageReadMode.RGB)
        img = img.transpose(2, 0)
        img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        return img_ori, img1, img2, label, TL, SL, idx

    def get_labels(self):
        labels = []
        for i in range(len(self.filenames)):
            labels.append(self.id_dict[self.filenames[i].split('\\')[-3]])
        return labels

    def generate_TL_SL(self):
        random.seed(self.seed)
        n = len(self.labels)
        c = max(self.labels) + 1
        new_TL = copy.deepcopy(self.labels)
        new_SL = torch.zeros(n, c)

        avgC = 0
        partial_rate = self.rate_partial

        S = int(c * partial_rate)

        print("rate:{}".format(partial_rate))
        for i in range(n):

            # fixed num
            SL = random.sample(range(c), S)
            SL = np.sum(np.eye(c)[SL], axis=0)

            SL = torch.from_numpy(SL)
            while torch.sum(SL) == 1:
                r_m = random.sample(range(c), S)
                for j in range(len(r_m)):
                    SL[r_m[j]] = 1
            avgC += torch.sum(SL)
            if SL[self.labels[i]] == 1:
                new_SL[i] = torch.from_numpy(np.zeros(c))
            else:
                new_TL[i] = c
                new_SL[i] = SL

        avgC = avgC / n
        new_TL_array = np.array(new_TL)
        new_SL_num = len(new_TL_array[np.where(new_TL_array == c)])
        new_TL_num = n - new_SL_num
        new_SL_num /= n
        new_TL_num /= n
        print("Finish Generating Stochastic Label Sets. TL:{:.2f}, SL:{:.2f}, each_SL_num:{}!\n"
              .format(new_TL_num, new_SL_num, avgC))
        new_SL = new_SL.cpu().numpy()
        return new_TL, new_SL


class TestTinyImageNetDataset(Dataset):
    def __init__(self, id):
        self.filenames = glob.glob("D:\code\python\weakly_supervised\MCTF_SL/data/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = Compose([
            Resize(32),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('D:\code\python\weakly_supervised\MCTF_SL/data/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = read_image(img_path)
        if img.shape[0] == 1:
            img = read_image(img_path, ImageReadMode.RGB)
        img = img.transpose(2, 0)
        img = Image.fromarray(np.array(img))
        label = self.cls_dic[img_path.split('\\')[-1]]
        if self.transform:
            img = self.transform(img)
        return img, label

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, QMNIST
from augment.cutout import Cutout
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image
import random
import copy

class CombinedMNIST(Dataset):
    def __init__(self, datasets, rate_partial=0.3, seed=2):
        self.lengths = 0

        for i in range(len(datasets)):
            if i == 0:
                self.data = datasets[i].data
                self.targets = datasets[i].targets
            else:
                target_list = datasets[i].targets + 10 * i
                self.data = torch.cat([self.data, datasets[i].data], dim=0)
                self.targets = torch.cat([self.targets, target_list], dim=0)
            self.lengths = self.lengths + len(datasets[i])

        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(28, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.1307], std=[0.3081]),
        ])

        self.transform1 = Compose([
            RandomHorizontalFlip(),
            RandomCrop(28, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            ToTensor(),
            Normalize(mean=[0.1307], std=[0.3081]),
        ])

        self.seed = seed
        self.rate_partial = rate_partial
        self.TL, self.SL = self.generate_TL_SL()

    def __getitem__(self, index):

        img, target, TL, SL = self.data[index], self.targets[index], self.TL[index], self.SL[index]
        img = Image.fromarray(img.numpy(), mode='L')

        img_ori = self.transform(img)
        img1 = self.transform1(img)
        img2 = self.transform1(img)

        return img_ori, img1, img2, target, TL, SL, index

    def __len__(self):
        return self.lengths


    def generate_TL_SL(self):
        random.seed(self.seed)
        n = len(self.targets)
        c = max(self.targets).item() + 1
        new_TL = copy.deepcopy(self.targets)
        # new_SL = binarize_class(self.targets)
        new_SL = torch.zeros(n, c)

        avgC = 0
        partial_rate = self.rate_partial

        S = int(c * partial_rate)

        print("rate:{}".format(partial_rate))
        for i in range(n):

            # # stochastic num
            # SL = np.random.binomial(1, partial_rate, c)

            # fixed num
            SL = random.sample(range(c), S)
            SL = np.sum(np.eye(c)[SL], axis=0)

            SL = torch.from_numpy(SL)
            while torch.sum(SL) == 1:
                r_m = random.sample(range(c), S)
                for j in range(len(r_m)):
                    SL[r_m[j]] = 1
            avgC += torch.sum(SL)
            if SL[self.targets[i]] == 1:
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
        new_TL = new_TL.cpu().numpy()
        return new_TL, new_SL





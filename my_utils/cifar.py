import os
import os.path
import pickle
import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage

from augment.autoaugment_extra import CIFAR10Policy
from augment.cutout import Cutout

class MY_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, rate_partial=0.3, seed=2):

        super(MY_CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        # print(len(self.targets))
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform1 = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            CIFAR10Policy(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.rate_partial = rate_partial
        self.seed = seed
        self.TL, self.SL = self.generate_TL_SL

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, TL, SL = self.data[index], self.targets[index], self.TL[index], self.SL[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ori, img1, img2, target, TL, SL, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    @property
    def generate_TL_SL(self):
        random.seed(self.seed)
        n = len(self.targets)
        c = max(self.targets) + 1
        new_TL = self.targets
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
        return new_TL, new_SL




class MY_CIFAR100(MY_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        This is a subclass of the `CIFAR10` Dataset.
        """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

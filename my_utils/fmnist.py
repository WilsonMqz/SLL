import codecs
import errno
import os
import os.path
import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage

from augment.cutout import Cutout


class MY_FMNIST(VisionDataset):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'

    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, rate_partial=0.3, seed=2):

        super(MY_FMNIST, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.training_file
        else:
            downloaded_list = self.test_file

        self.data, self.targets = torch.load(
            os.path.join(self.root, self.processed_folder, downloaded_list))

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

        self.rate_partial = rate_partial
        self.seed = seed
        self.TL, self.SL = self.generate_TL_SL()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, TL, SL = self.data[index], self.targets[index], self.TL[index], self.SL[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ori, img1, img2, target, TL, SL, index

    def __len__(self):
        return len(self.data)

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

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Fashion-MNIST data if it doesn't exist in processed_folder already."""
        import urllib.request
        import gzip

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

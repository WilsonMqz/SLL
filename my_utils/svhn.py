import os
import os.path
import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage

from augment.autoaugment_extra import SVHNPolicy
from augment.cutout import Cutout


class MY_SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, rate_partial=0.3, seed=2,
    ) -> None:
        super(MY_SVHN, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.targets = self.labels

        self.rate_partial = rate_partial
        self.seed = seed
        self.TL, self.SL = self.generate_TL_SL()

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transform1 = Compose([
            ToTensor(),
            Cutout(n_holes=1, length=20),
            ToPILImage(),
            SVHNPolicy(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, TL, SL = self.data[index], self.targets[index], self.TL[index], self.SL[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ori, img1, img2, target, TL, SL, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

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

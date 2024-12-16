import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from my_utils.cifar import MY_CIFAR10, MY_CIFAR100
from my_utils.combine_mnist import CombinedMNIST
from my_utils.tiny import TrainTinyImageNetDataset, TestTinyImageNetDataset
from my_utils.fmnist import MY_FMNIST
from my_utils.kmnist import MY_KMNIST
from my_utils.svhn import MY_SVHN

def cifar100_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    cifar100_train_ds = MY_CIFAR100(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(cifar100_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'
          .format(datasets.CIFAR100.__name__, len(cifar100_train_ds), 100))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Data loader for test dataset
    cifar100_test_ds = datasets.CIFAR100(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar100_test_ds)))
    test_loader = DataLoader(cifar100_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader


def combine_mnist_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    transform = Compose([
        ToTensor(),
        Normalize((0.5), (0.5)),
    ])

    train_mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    train_fashion_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    train_kmnist_dataset = datasets.KMNIST(root=data_dir, train=True, download=True)

    # Combine train datasets
    train_datasets_list = [train_mnist_dataset, train_fashion_dataset, train_kmnist_dataset]
    train_dataset = CombinedMNIST(datasets=train_datasets_list, rate_partial=rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)

    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'
          .format(datasets.FashionMNIST.__name__, len(train_dataset), 30))

    # Load test datasets
    test_mnist_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    test_fashion_dataset = datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
    test_kmnist_dataset = datasets.KMNIST(root=data_dir, train=False, transform=transform, download=True)

    # Combine test datasets
    test_datasets_list = [test_mnist_dataset, test_fashion_dataset, test_kmnist_dataset]
    test_dataset = CombinedMNIST(test_datasets_list)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)

    return train_loader, test_loader


def tiny_imagenet_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    id_dict = {}
    for i, line in enumerate(open('D:\code\python\weakly_supervised\MCTF_SL/data/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    train_dataset = TrainTinyImageNetDataset(id=id_dict, rate_partial=rate)
    test_dataset = TestTinyImageNetDataset(id=id_dict)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def fashion_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    fmnist_train_ds = MY_FMNIST(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(fmnist_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'
          .format(datasets.FashionMNIST.__name__, len(fmnist_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.1307), (0.3081)),
    ])
    # Data loader for test dataset
    fmnist_test_ds = datasets.FashionMNIST(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(fmnist_test_ds)))
    test = DataLoader(fmnist_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test


def kuzushiji_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    kmnist_train_ds = MY_KMNIST(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(kmnist_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, multiprocessing_context='spawn')
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'
          .format(datasets.KMNIST.__name__, len(kmnist_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.5), (0.5)),
    ])
    # Data loader for test dataset
    kmnist_test_ds = datasets.KMNIST(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(kmnist_test_ds)))
    test = DataLoader(kmnist_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, multiprocessing_context='spawn')
    return train_loader, test


def cifar10_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    cifar10_train_ds = MY_CIFAR10(data_dir, train=True, download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(cifar10_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'
          .format(datasets.CIFAR10.__name__, len(cifar10_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test_loader = DataLoader(cifar10_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def svhn_dataloaders(data_dir, rate, batch_size, num_workers):
    print('Data Preparation')
    svhn_train_ds = MY_SVHN(data_dir, split='train', download=True, rate_partial=rate)
    train_loader = torch.utils.data.DataLoader(svhn_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.SVHN.__name__,
                                                                                          len(svhn_train_ds), 10))
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Data loader for test dataset
    svhn_test_ds = datasets.SVHN(data_dir, transform=test_transform, split='test', download=True)
    print('Test set -- Num_samples: {0}'.format(len(svhn_test_ds)))
    test_loader = DataLoader(svhn_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader




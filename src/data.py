import torch 
import torchvision 
import torchvision.transforms as transforms
import os, random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pickle 
from torch.utils.data.distributed import DistributedSampler

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

def get_transforms(image_size, data_name = "cifar10", algo='supcon'):
    if data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif data_name == "tinyimagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # for solarization 
    solarize_algo = ["byol", "vicreg", "bt"]
    solarize_p = 0.0
    if any([i in algo for i in solarize_algo]):
        solarize_p = 0.1
    
    s = 0.5

    # for smaller image datasets, no gaussian blur 
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        transforms.RandomGrayscale(p = 0.2),
        Solarization(p = 0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std=std)
    ])

    train_transforms_prime = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        transforms.RandomGrayscale(p = 0.2),
        Solarization(p = solarize_p),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std=std)
    ])

    train_transforms_mlp = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = mean, std = std)])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])

    return {"train_transforms": train_transforms, 
            "train_transforms_prime": train_transforms_prime, 
            "train_transforms_mlp": train_transforms_mlp, 
            "test_transforms": test_transforms}

class DataCifar():
    def __init__(self, algo = "simclr", data_name = "cifar10", data_dir = "datasets/cifar10", target_transforms = {}):
        if data_name == "cifar10":
            self.data = torchvision.datasets.CIFAR10(data_dir, train = True, download = True)
        elif data_name == "cifar100":
            self.data = torchvision.datasets.CIFAR100(data_dir, train = True, download = True)

        self.algo = algo
        self.target_transform = target_transform.get("train_transforms", None)
        self.target_transform_prime = target_transform.get("train_transforms_prime", None)
    
        if self.algo == "triplet":
            len_data = len(self.data)
            data_classes = len(self.data.classes)
            self.all_data = {i:[] for i in range(data_classes)}

            for idx in range(len_data):
                self.all_data[self.data[idx][1]].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.algo == "triplet":
            anc, anc_label = self.data[idx]

            pos_idx = random.choice(self.all_data[anc_label])

            all_classes_idx = list(self.all_data.keys())
            all_classes_idx.remove(anc_label)
            neg_label = random.choice(all_classes_idx)

            neg_idx = random.choice(self.all_data[neg_label])

            pos, pos_label = self.data[pos_idx]
            neg, neg_label = self.data[neg_idx]

            anc = self.target_transform(anc)
            pos = self.target_transform(pos)
            neg = self.target_transform(neg)

            return anc, anc_label, pos, pos_label, neg, neg_label
            
        image, label = self.data[idx]

        img1 = self.target_transform(image)
        img2 = self.target_transform_prime(image)
        return img1, img2, label 
        

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    all_transforms = get_transforms(image_size, data_name = "cifar100", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']


    train_transforms = {"train_transforms": all_transforms["train_transforms"],
                        "train_transforms_prime": all_transforms["train_transforms_prime"]}

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar100", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR100(
        data_dir,
        transform = all_transforms["train_transforms_mlp"],
        train = True,
        download = True
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform= all_transforms["test_transforms"],
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset

def Cifar10DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    all_transforms = get_transforms(image_size, data_name = "cifar10", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_transforms = {"train_transforms": all_transforms["train_transforms"],
                        "train_transforms_prime": all_transforms["train_transforms_prime"]}

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar10", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR10(
        data_dir,
        transform = all_transforms["train_transforms_mlp"],
        train = True,
        download = True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        data_dir, 
        transform= all_transforms["test_transforms"],
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset
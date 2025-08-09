from custom_dataset import *
import numpy as np
from torchvision.datasets.folder import default_loader
import os
import json

from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from loguru import logger
import argparse
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
import pandas as pd


def get_dataset_and_dict(preprocess, args):
    if args.dataset_name == "cifar10":
        dataset = CustomCIFAR10("/home/ps/_lzj/GCD/dataset/cifar10/", transform=preprocess, train=True)
        class_names_cifar10 = np.load("/home/ps/_lzj/GCD/dataset/cifar10/cifar-10-batches-py/batches.meta",
                                      allow_pickle=True)
        class_names_arr = class_names_cifar10["label_names"]  # 10个分类名
        print(f'class_names_arr:{class_names_arr}')
        class_names_dict = {}
        for cid, class_name in enumerate(class_names_arr):
            class_names_dict[cid] = class_name

    elif args.dataset_name == "cifar100":
        dataset = CustomCIFAR100("/home/ps/_jinwei/Dataset/CIFAR/cifar100", transform=preprocess, train=True)
        class_names_cifar100 = np.load("/home/ps/_jinwei/Dataset/CIFAR/cifar100/cifar-100-python/meta", allow_pickle=True)
        fine_label_names = class_names_cifar100["fine_label_names"]  # 100个细粒度分类名
        coarse_label_names = class_names_cifar100["coarse_label_names"]  # 20个粗粒度分类名

        print(f'class_names_arr:{fine_label_names}')
        # print(f'class_names_coars:{coarse_label_names}')

        class_names_dict = {"fine": {}, "coarse": {}}
        for cid, fine_class_name in enumerate(fine_label_names):
            class_names_dict["fine"][cid] = fine_class_name
        for cid, coarse_class_name in enumerate(coarse_label_names):
            class_names_dict["coarse"][cid] = coarse_class_name

        print(f'class_names_dict:{class_names_dict}')






    elif args.dataset_name == "imagenet_100":
        imagenet_root = '/home/ps/_lzj/GCD/dataset/ImageNet100/ILSVRC12'
        dataset = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=preprocess)
        class_to_idx_map = dataset.class_to_idx  # 这个map的key是imagenet100的编号，values是下标(0-99)

        labels_path = os.path.join(imagenet_root, 'Labels.json')
        with open(labels_path, 'r') as file:
            class_names_imagenet_dict = json.load(file)
            # 返回一个字典，imagenet100的key是类的编号，values是真正的类名
        file.close()
        '''
        用class_names_imagenet_dict与class_to_idx_map
        才能得到一个类名字与id对应的map
        '''
        class_names_dict = {}  # 这个map的key是下标(0-99)，values是类的真正名字，不是imagenet100的编号
        for class_order in class_names_imagenet_dict.keys():
            real_class_name = class_names_imagenet_dict[class_order]
            class_id = class_to_idx_map[class_order]
            class_names_dict[class_id] = real_class_name

    elif args.dataset_name == "cub":
        dataset = CustomCub2011("/home/ps/_jinwei/Dataset/CUB/", transform=preprocess, train=True)
        class_names_dict = {}
        with open(f"/home/ps/_jinwei/Dataset/CUB/CUB_200_2011/classes.txt", 'r') as txt_file:
            for i, txt in enumerate(txt_file):
                txt = txt.replace("\n", "")
                class_name = txt[txt.find(".") + 1:]
                class_names_dict[i] = class_name
                '''
                虽然cub的class的txt里面从1开始，但是实际从dataset里面取出来，label的下标还是0-199
                '''
        txt_file.close()

    elif args.dataset_name == "aircraft":
        dataset = FGVCAircraft("/home/ps/_jinwei/Dataset/Aircraft/fgvc-aircraft-2013b/", transform=preprocess,
                               split="trainval")
        class_names_dict = {}
        with open(f"/home/ps/_jinwei/Dataset/Aircraft/fgvc-aircraft-2013b/data/variants.txt", 'r') as txt_file:
            for i, txt in enumerate(txt_file):
                txt = txt.replace("\n", "")
                class_name = txt
                class_names_dict[i] = class_name
                '''
                虽然cub的class的txt里面从1开始，但是实际从dataset里面取出来，label的下标还是0-199
                '''
        txt_file.close()

    elif args.dataset_name == "scars":
        dataset = CarsDataset(transform=preprocess, train=True)
        class_names_dict = {}
        with open(f"/home/ps/_jinwei/Dataset/Stanford_cars/label_map.txt", 'r') as txt_file:
            for i, txt in enumerate(txt_file):
                txt = txt.replace("\n", "")
                class_name = txt
                class_names_dict[i] = class_name
                '''
                虽然cub的class的txt里面从1开始，但是实际从dataset里面取出来，label的下标还是0-199
                '''
        txt_file.close()

    return dataset, class_names_dict
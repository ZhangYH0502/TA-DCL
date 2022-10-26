import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from PIL import Image

from dataloaders.chestmnist_dataset import ChestmnistDataset

import warnings
warnings.filterwarnings("ignore")


def get_data(args):
    dataset = args.dataset
    data_root = args.dataroot

    rescale = args.scale_size

    normTransform = transforms.Normalize(mean=[0.5], std=[0.5])
    scale_size = rescale
    
    trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomHorizontalFlip(),
                                        normTransform
    ])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        normTransform
    ])

    test_dataset = None
    test_loader = None

    chestmnist_root = os.path.join(data_root, 'chestmnist')

    train_dataset = ChestmnistDataset(
        split='train',
        data_file=os.path.join(chestmnist_root, 'chestmnist.npz'),
        transform=trainTransform)
    valid_dataset = ChestmnistDataset(
        split='val',
        data_file=os.path.join(chestmnist_root, 'chestmnist.npz'),
        transform=testTransform)
    test_dataset = ChestmnistDataset(
        split='test',
        data_file=os.path.join(chestmnist_root, 'chestmnist.npz'),
        transform=testTransform)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  drop_last=False)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=args.workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    return train_loader, valid_loader, test_loader


def get_random_neg_samples(num=20):
    normTransform = transforms.Normalize(mean=[0.5], std=[0.5])
    Transform_h = transforms.Compose([transforms.Resize((640, 640)),
                                      transforms.RandomHorizontalFlip(),
                                      normTransform])

    data_file = "data.npz"
    root = "dataset/"
    split_data = np.load(data_file)
    imgs_h = split_data['train_images_healthy']
    labs_h = split_data['train_labels_healthy']

    num_h = len(imgs_h)

    idx_list = np.arange(num_h, dtype=int)
    random.shuffle(idx_list)

    images_batch = []
    labels_batch = []

    for i in range(num):

        image = Image.open(root + "/" + imgs_h[idx_list[i]])
        image = torch.Tensor(np.array(image))
        if len(image.shape) > 2:
            image = image[:, :, 0]
        image = Transform_h(image)

        labels = labs_h[idx_list[i]].astype(int)
        labels = torch.Tensor(labels)

        images_batch.append(image)
        labels_batch.append(labels)

    images_batch = torch.stack(images_batch, dim=0)
    images_batch = images_batch.view(images_batch[0], 1, images_batch[1], images_batch[2])

    labels_batch = torch.stack(labels_batch, dim=0)

    return images_batch, labels_batch

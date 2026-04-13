import os
import torch
import numpy as np
import random
from torchvision import transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(device_id):
    return torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

def tensor_to_numpy(x):
    return x.cpu().detach().numpy()


def get_transforms(img_size=224, train=True):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
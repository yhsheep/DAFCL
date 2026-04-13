import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from PIL import Image
import pickle
import torchvision
from utils.toolkit import get_transforms 

class SimpleDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2: 
                img = Image.fromarray(img, mode='L').convert('RGB')
            else:
                img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class ContinualDataManager:
    def __init__(self, args):
        self.args = args
        self.root = args.data_path
        self.img_size = args.img_size
        self.num_clients = args.num_clients
        self.beta = getattr(args, 'non_iid_beta', 0.5)

        self.train_data, self.train_targets, self.test_data, self.test_targets = self._load_dataset()

        num_train = len(self.train_data)
        indices = np.arange(num_train)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        
        proxy_split = int(num_train * 0.01) 
        self.proxy_indices = indices[:proxy_split]
        self.private_indices = indices

        self.num_classes = args.num_classes
        self.class_order = np.arange(self.num_classes)
        if args.class_order == 'random':
            np.random.shuffle(self.class_order)
        
        self.tasks = []
        classes_per_task = self.num_classes // args.num_tasks
        for t in range(args.num_tasks):
            self.tasks.append(self.class_order[t * classes_per_task : (t+1) * classes_per_task])
            
        self.client_task_indices = self._partition_data_dirichlet()

    def _load_dataset(self):
        dataset_name = self.args.dataset.lower()
        
        if dataset_name == 'cifar100':
            train_data, train_targets = self._load_cifar100_batch(train=True)
            test_data, test_targets = self._load_cifar100_batch(train=False)
            
        elif dataset_name == 'cifar10':
            train_ds = torchvision.datasets.CIFAR10(self.root, train=True, download=True)
            test_ds = torchvision.datasets.CIFAR10(self.root, train=False, download=True)
            train_data, train_targets = train_ds.data, np.array(train_ds.targets)
            test_data, test_targets = test_ds.data, np.array(test_ds.targets)
            
        elif dataset_name == 'mnist':
            train_ds = torchvision.datasets.MNIST(self.root, train=True, download=True)
            test_ds = torchvision.datasets.MNIST(self.root, train=False, download=True)
            # MNIST 是 (N, 28, 28)，SimpleDataset 会处理转换
            train_data, train_targets = train_ds.data.numpy(), np.array(train_ds.targets)
            test_data, test_targets = test_ds.data.numpy(), np.array(test_ds.targets)
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        return train_data, train_targets, test_data, test_targets

    def _load_cifar100_batch(self, train=True):
        file_name = 'train' if train else 'test'
        file_path = os.path.join(self.root, 'cifar-100-python', file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        data = entry['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        return data.astype('uint8'), np.array(entry['fine_labels'])

    def _partition_data_dirichlet(self):
        np.random.seed(self.args.seed)
        client_task_indices = {c: {t: [] for t in range(self.args.num_tasks)} for c in range(self.num_clients)}
        
        for t_id, task_classes in enumerate(self.tasks):
            for cls in task_classes:
                idx_in_full = np.where(self.train_targets == cls)[0]
                idx = idx_in_full.copy()
                np.random.shuffle(idx)
                
                proportions = np.random.dirichlet([self.beta] * self.num_clients)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
                
                split_indices = np.split(idx, proportions)
                for c_id, client_idx in enumerate(split_indices):
                    client_task_indices[c_id][t_id].extend(client_idx.tolist())
        return client_task_indices

    def get_proxy_loader(self, batch_size=128):
        transform = get_transforms(self.img_size, train=False)
        
        cifar_proxy_data = self.train_data[self.proxy_indices]
        cifar_proxy_targets = self.train_targets[self.proxy_indices]
        cifar_dataset = SimpleDataset(cifar_proxy_data, cifar_proxy_targets, transform=transform)

        tiny_path = os.path.join(self.args.data_path, 'tiny-imagenet-200', 'train')
        if os.path.exists(tiny_path):
            from torchvision.datasets import ImageFolder
            tiny_dataset = ImageFolder(tiny_path, transform=transform)
            combined_dataset = ConcatDataset([cifar_dataset, tiny_dataset])
            print(f"[Data] Mixed Proxy Dataset created for {self.args.dataset}")
        else:
            combined_dataset = cifar_dataset
            print(f"[Data] Using {self.args.dataset} Subset as proxy only.")

        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def get_task_loader(self, task_id, client_id=None, mode='train', batch_size=None):
        if batch_size is None: batch_size = self.args.batch_size
        transform = get_transforms(self.img_size, train=(mode == 'train'))

        if mode == 'train':
            indices = self.client_task_indices[client_id][task_id]
            data = self.train_data[indices]
            targets = self.train_targets[indices]
        elif mode == 'test_current':
            target_classes = self.tasks[task_id]
            mask = np.isin(self.test_targets, target_classes)
            data, targets = self.test_data[mask], self.test_targets[mask]
        else:
            target_classes = np.concatenate([self.tasks[i] for i in range(task_id + 1)])
            mask = np.isin(self.test_targets, target_classes)
            data, targets = self.test_data[mask], self.test_targets[mask]

        dataset = SimpleDataset(data, targets, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=self.args.num_workers)
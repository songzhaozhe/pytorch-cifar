from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
# import torch.utils.data as data
# from .utils import download_url, check_integrity

erase_start = 99

ERASE_LABELS = [i for i in range(erase_start, 100)]

class custom_CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            new_train = np.zeros([50000-len(ERASE_LABELS)*500, 32, 32, 3], dtype=np.uint8) # uint8 is necessary
            new_labels = []
            idx = 0
            for i in range(len(self.train_labels)):
                if not (self.train_labels[i] in ERASE_LABELS):
                    new_train[idx] = self.train_data[i]
                    new_labels.append(self.train_labels[i])
                    idx += 1
            self.train_data = new_train
            self.train_labels = new_labels
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            new_test = np.zeros([10000-len(ERASE_LABELS)*100, 32, 32, 3], dtype=np.uint8) # uint8 is necessary
            new_labels = []
            idx = 0
            for i in range(len(self.test_labels)):
                if not (self.test_labels[i] in ERASE_LABELS):
                    new_test[idx] = self.test_data[i]
                    new_labels.append(self.test_labels[i])
                    idx += 1
            self.test_data = new_test
            self.test_labels = new_labels
    def __len__(self):
        if self.train:
            return 50000-len(ERASE_LABELS)*500
        else:
            return 10000-len(ERASE_LABELS)*100

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
import numpy
# import torch.utils.data as data
# from .utils import download_url, check_integrity

erase_start = 100

ERASE_LABELS = [i for i in range(erase_start, 100)]

class pro_CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, NEW_LABEL_START, proportion, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.tot = (int)(500*proportion)
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
            self.train_labels = np.asarray(self.train_labels)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            perm = np.random.permutation(self.train_data.shape[0])
            self.train_data = self.train_data[perm]
            self.train_labels = self.train_labels[perm]

            tot = self.tot
            cnt = [0 for i in range(NEW_LABEL_START)]
            new_train = np.zeros([tot*NEW_LABEL_START + 500 * (100-NEW_LABEL_START), 32, 32, 3], dtype=np.uint8) # uint8 is necessary
            self.size = tot*NEW_LABEL_START + 500 * (100-NEW_LABEL_START)
            new_labels = []
            idx = 0
            for i in range(len(self.train_labels)):
                if self.train_labels[i] >= NEW_LABEL_START:
                    new_train[idx] = self.train_data[i]
                    new_labels.append(self.train_labels[i])
                    idx += 1
                elif (cnt[self.train_labels[i]] < tot):
                    cnt[self.train_labels[i]] += 1
                    new_train[idx] = self.train_data[i]
                    new_labels.append(self.train_labels[i])
                    idx += 1
            self.train_data = new_train
            self.train_labels = new_labels
        else:
            raise NotImplementedError
    def __len__(self):
        if self.train:
            return self.size
        else:
            return 10000-len(ERASE_LABELS)*100

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

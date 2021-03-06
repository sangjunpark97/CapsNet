# -*- coding: utf-8 -*-
"""capsnet_final.ipynb의 사본

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18OrD9dsMmP4SvUl2lUyewvCLwkfp0Fvy
"""





import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
!pip install pyyaml h5py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np

class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=9, # fixme constant
                               stride=2,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))

from google.colab import drive
drive.mount('/content/drive')

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1) # num_units = 출력 개수로 본문에서는 10개를 사용한다.

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)
        # in_unit = 6*6*32
        # feature = 8이랑 10
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2) # transpose(1열, 2열): 1열과 2열의 자리를 바꿈(0부터 시작작)
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4) # 
        # x에 10 곱함 -> dim2로 인해 x를 (잘 모르겠음음)
        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)# (1152  10  8  16)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij)
            
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            
            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)
            
            # [ 10  16  ]
            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            
            # [  ]
            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            
            #  [  1152   10   ]
            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F



class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleNetwork, self).__init__()

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                      out_channels=conv_outputs)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)

        reconstruction_size = image_width * image_height * image_channels
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size()))
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error





from __future__ import print_function
import os
import errno
import struct

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


class SmallNORB(data.Dataset):
    """`MNIST <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.
    Args:
        root (string): Root directory of dataset where processed folder and
            and  raw folder exist.
        train (bool, optional): If True, creates dataset from the training files,
            otherwise from the test files.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If the dataset is already processed, it is not processed
            and downloaded again. If dataset is only already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        info_transform (callable, optional): A function/transform that takes in the
            info and transforms it.
        mode (string, optional): Denotes how the images in the data files are returned. Possible values:
            - all (default): both left and right are included separately.
            - stereo: left and right images are included as corresponding pairs.
            - left: only the left images are included.
            - right: only the right images are included.
    """

    dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    data_files = {
        'train': {
            'dat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
                "md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
                "md5": "8138a0902307b32dfa0025a36dfa45ec"
            },
            'info': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
                "md5_gz": "51dee1210a742582ff607dfd94e332e3",
                "md5": "19faee774120001fc7e17980d6960451"
            },
            'cat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
                "md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
        'test': {
            'dat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
                "md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
                "md5": "e9920b7f7b2869a8f1a12e945b2c166c"
            },
            'info': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
                "md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
                "md5": "7c5b871cc69dcadec1bf6a18141f5edc"
            },
            'cat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
                "md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
    }

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_image_file = 'train_img'
    train_label_file = 'train_label'
    train_info_file = 'train_info'
    test_image_file = 'test_img'
    test_label_file = 'test_label'
    test_info_file = 'test_info'
    extension = '.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, info_transform=None, download=True,
                 mode="all"):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.info_transform = info_transform
        self.train = train  # training set or test set
        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # load test or train set
        image_file = self.train_image_file if self.train else self.test_image_file
        label_file = self.train_label_file if self.train else self.test_label_file
        info_file = self.train_info_file if self.train else self.test_info_file

        # load labels
        self.labels = self._load(label_file)

        # load info files
        self.infos = self._load(info_file)

        # load right set
        if self.mode == "left":
            self.data = self._load("{}_left".format(image_file))

        # load left set
        elif self.mode == "right":
            self.data = self._load("{}_right".format(image_file))

        elif self.mode == "all" or self.mode == "stereo":
            left_data = self._load("{}_left".format(image_file))
            right_data = self._load("{}_right".format(image_file))

            # load stereo
            if self.mode == "stereo":
                self.data = torch.stack((left_data, right_data), dim=1)

            # load all
            else:
                self.data = torch.cat((left_data, right_data), dim=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            mode ``all'', ``left'', ``right'':
                tuple: (image, target, info)
            mode ``stereo'':
                tuple: (image left, image right, target, info)
        """
        target = self.labels[index % 24300] if self.mode is "all" else self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        info = self.infos[index % 24300] if self.mode is "all" else self.infos[index]
        if self.info_transform is not None:
            info = self.info_transform(info)

        if self.mode == "stereo":
            img_left = self._transform(self.data[index, 0])
            img_right = self._transform(self.data[index, 1])
            return img_left, img_right, target, info

        img = self._transform(self.data[index])
        return img, target

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        # doing this so that it is consistent with all other data sets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load(self, file_name):
        return torch.load(os.path.join(self.root, self.processed_folder, file_name + self.extension))

    def _save(self, file, file_name):
        with open(os.path.join(self.root, self.processed_folder, file_name + self.extension), 'wb') as f:
            torch.save(file, f)

    def _check_exists(self):
        """ Check if processed files exists."""
        files = (
            "{}_left".format(self.train_image_file),
            "{}_right".format(self.train_image_file),
            "{}_left".format(self.test_image_file),
            "{}_right".format(self.test_image_file),
            self.test_label_file,
            self.train_label_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, f + self.extension)) for f in files]
        return False not in fpaths

    def _flat_data_files(self):
        return [j for i in self.data_files.values() for j in list(i.values())]

    def _check_integrity(self):
        """Check if unpacked files have correct md5 sum."""
        root = self.root
        for file_dict in self._flat_data_files():
            filename = file_dict["name"]
            md5 = file_dict["md5"]
            fpath = os.path.join(root, self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # check if already extracted and verified
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            # download and extract
            for file_dict in self._flat_data_files():
                url = self.dataset_root + file_dict["name"] + '.gz'
                filename = file_dict["name"]
                gz_filename = filename + '.gz'
                md5 = file_dict["md5_gz"]
                fpath = os.path.join(self.root, self.raw_folder, filename)
                gz_fpath = fpath + '.gz'

                # download if compressed file not exists and verified
                download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

                print('# Extracting data {}\n'.format(filename))

                with open(fpath, 'wb') as out_f, \
                        gzip.GzipFile(gz_fpath) as zip_f:
                    out_f.write(zip_f.read())

                os.unlink(gz_fpath)

        # process and save as torch files
        print('Processing...')

        # create processed folder
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # read train files
        left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
        train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
        train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

        # read test files
        left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
        test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
        test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

        # save training files
        self._save(left_train_img, "{}_left".format(self.train_image_file))
        self._save(right_train_img, "{}_right".format(self.train_image_file))
        self._save(train_label, self.train_label_file)
        self._save(train_info, self.train_info_file)

        # save test files
        self._save(left_test_img, "{}_left".format(self.test_image_file))
        self._save(right_test_img, "{}_right".format(self.test_image_file))
        self._save(test_label, self.test_label_file)
        self._save(test_info, self.test_info_file)

        print('Done!')

    @staticmethod
    def _parse_header(file_pointer):
        # Read magic number and ignore
        struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        return dimensions

    def _read_image_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300, 2, 96, 96]
            num_samples, _, height, width = dimensions

            left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
            right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

            for i in range(num_samples):

                # left and right images stored in pairs, left first
                left_samples[i, :, :] = self._read_image(f, height, width)
                right_samples[i, :, :] = self._read_image(f, height, width)

        return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

    @staticmethod
    def _read_image(file_pointer, height, width):
        """Read raw image data and restore shape as appropriate. """
        image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
        image = np.uint8(np.reshape(image, newshape=(height, width)))
        return image

    def _read_label_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300]
            num_samples = dimensions[0]

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            labels = np.zeros(shape=num_samples, dtype=np.int32)
            for i in range(num_samples):
                category, = struct.unpack('<i', f.read(4))
                labels[i] = category
            return torch.LongTensor(labels)

    def _read_info_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:

            dimensions = self._parse_header(f)
            assert dimensions == [24300, 4]
            num_samples, num_info = dimensions

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

            for r in range(num_samples):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    infos[r, c] = info

        return torch.LongTensor(infos)

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset):
  # Compute validation split
  train_size = len(train_dataset)
  indices = list(range(train_size))
  split = int(np.floor(valid_size * train_size))
  np.random.shuffle(indices)
  train_idx = indices[split:]
  train_sampler = SubsetRandomSampler(train_idx)
  #valid_sampler = SubsetRandomSampler(valid_idx)
  
  # Create dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)
  #valid_loader = torch.utils.data.DataLoader(valid_dataset,
  #                                           batch_size=batch_size,
  #                                           sampler=valid_sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
  return train_loader, test_loader


def load_small_norb(batch_size):
    path = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    train_transform = transforms.Compose([
                          transforms.Resize(48),
                          #transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize((0.0,), (0.3081,))
                      ])
    valid_transform = transforms.Compose([
                          transforms.Resize(48),
                          #transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    test_transform = transforms.Compose([
                          transforms.Resize(48),
                          #transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    
    train_dataset = SmallNORB(path, train=True, download=True, transform=train_transform)
    valid_dataset = SmallNORB(path, train=True, download=True, transform=valid_transform)
    test_dataset = SmallNORB(path, train=False, transform=test_transform)
    valid_size = 0 #DEFAULT_VALIDATION_SIZE 
    return build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset)

train_loader, test_loader=load_small_norb(16)

learning_rate = 0.0005
batch_size = 16
test_batch_size = 16

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

#
# Load MNIST dataset.
#

# Normalization for MNIST dataset.
dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

#train_dataset = datasets.MNIST('C:/Users/Jet Zhang/Desktop/pytorch/GAN/mnist', train=True, download=True, transform=dataset_transform)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#test_dataset = datasets.MNIST('C:/Users/Jet Zhang/Desktop/pytorch/GAN/mnist', train=False, download=True, transform=dataset_transform)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


#
# Create capsule network.
#

conv_inputs = 1
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 32 * 6 * 6  # fixme get from conv2d
output_unit_size = 16

network = CapsuleNetwork(image_width=48,
                         image_height=48,
                         image_channels=1,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=5, # one for each MNIST digit
                         output_unit_size=output_unit_size)
network.load_state_dict(torch.load('/content/drive/MyDrive/model/epoch.pt'))
network.eval()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

import numpy as np
print(network.primary.state_dict().keys())
a=network.primary.state_dict()['unit_0.conv0.weight']
a=a.numpy()
print(a)
a=a.flatten()
#np.save('/content/drive/MyDrive/model/NORB_conv0_weight', a)

import struct

mydata=np.load('/content/drive/MyDrive/model/NORB_conv0_weight.npy')
mydata
f=open("/content/drive/MyDrive/model/NORB_conv0_weight.bin","wb")
myfmt='f'*len(mydata)
#  You can use 'd' for double and < or > to force endinness
bin=struct.pack(myfmt,*mydata)
print(bin)
f.write(bin)
f.close()

mydata=np.load('/content/drive/MyDrive/model/NORB_conv0_bias.npy')
mydata
f=open("/content/drive/MyDrive/model/NORB_conv0_bias.bin","wb")
myfmt='f'*len(mydata)
#  You can use 'd' for double and < or > to force endinness
bin=struct.pack(myfmt,*mydata)
print(bin)
f.write(bin)
f.close()

for data,target in test_loader:
    X=data[1]
    break
X=X.reshape(1,1,48,48)
print(X.shape)
Y=network.digits(network.primary(network.conv1(X)))

Y=torch.tensor(Y)
Y=Y.numpy()
print(Y.shape)
print(Y)
Y=Y.flatten()
#np.save('/content/drive/MyDrive/model/small_NORB', X)

mydata=np.load('/content/drive/MyDrive/model/small_NORB.npy')
mydata
f=open("/content/drive/MyDrive/model/small_NORB.bin","wb")
myfmt='f'*len(mydata)
#  You can use 'd' for double and < or > to force endinness
bin=struct.pack(myfmt,*mydata)
print(bin)
f.write(bin)
f.close()

# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

# This is the test function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def test():
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=network.digits.num_units)

        data, target = Variable(data, volatile=True), Variable(target_one_hot)

        output = network(data)

        test_loss += network.loss(data, output, target).item() # sum up batch loss

        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1].cpu()

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# This is the train function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def train(epoch):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    last_loss = None
    log_interval = 1
    network.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target_one_hot = to_one_hot(target, length=network.digits.num_units)
        #print(target)
        data, target = Variable(data), Variable(target_one_hot)
        #print(data.shape)
        optimizer.zero_grad()

        output = network(data)

        loss = network.loss(data, output, target)
        loss.backward()
        last_loss = loss.item()

        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),100. * batch_idx / len(train_loader),
                loss.item()))
        
    
    return last_loss

num_epochs = 30

for epoch in range(1, num_epochs + 1):
    with torch.no_grad():
      test()
    last_loss = train(epoch)
    torch.save(network.state_dict(), '/content/drive/MyDrive/model/epoch3.pt')

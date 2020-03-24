import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


def random_flip():
    pass


def random_crop():
    pass


class SKDataset(data.Dataset):
    def __init__(self, im_root, gt_root, resize, is_test=False):
        self.is_test = is_test
        self.resize = resize
        self.ims = [im_root + f for f in os.listdir(im_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.ims = sorted(self.ims)
        self.gts = sorted(self.gts)
        self.size = len(self.ims)

        self.im_tfm = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_tfm = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])
        self.index = 0

    def __getitem__(self, index):
        im = self.rgb_loader(self.ims[index])
        name = self.ims[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        if name.endswith('.png'):
            name = name.split('.png')[0]
        im_t = self.im_tfm(im)
        if self.is_test:
            return im_t, name

        gt = self.binary_loader(self.gts[index])

        if not self.is_test:
            gt_t = self.gt_tfm(gt)
        else:
            gt_t = transforms.ToTensor(gt)

        return im_t, gt_t, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            im = Image.open(f)
            return im.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            im = Image.open(f)
            return im.convert('L')

    def __len__(self):
        return self.size

    def load_data(self):
        im = self.rgb_loader(self.ims[self.index])
        im = self.im_tfm(im).unsqueeze(0)
        name = self.ims[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        if name.endswith('.png'):
            name = name.split('.png')[0]
        self.index += 1
        return im, name


def sk_loader(im_root, gt_root, batch_size, train_size, shuffle=True, num_worker=12, pin_memory=True):
    dataset = SKDataset(im_root, gt_root, train_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_worker,
                                  pin_memory=pin_memory)
    return data_loader


class SOD3DDataset(data.Dataset):
    def __init__(self, im_root, gt_root, dp_root, resize, is_test=False):
        self.is_test = is_test
        self.resize = resize
        self.ims = [im_root + f for f in os.listdir(im_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.dps = [dp_root + f for f in os.listdir(dp_root) if
                    f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
        self.ims = sorted(self.ims)
        self.gts = sorted(self.gts)
        self.dps = sorted(self.dps)
        self.size = len(self.ims)

        self.im_tfm = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_tfm = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])
        self.dp_tfm = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.index = 0

    def __getitem__(self, index):
        im = self.rgb_loader(self.ims[index])
        gt = self.binary_loader(self.gts[index])
        dp = self.rgb_loader(self.dps[index])
        im_t = self.im_tfm(im)
        dp_t = self.dp_tfm(dp)
        if not self.is_test:
            gt_t = self.gt_tfm(gt)
        else:
            gt_t = transforms.ToTensor(gt)
        name = self.ims[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        if name.endswith('.png'):
            name = name.split('.png')[0]
        return im_t, gt_t, dp_t, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            im = Image.open(f)
            return im.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            im = Image.open(f)
            return im.convert('L')

    def __len__(self):
        return self.size

    def load_data(self):
        im = self.rgb_loader(self.ims[self.index])
        im = self.im_tfm(im).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        dp = self.rgb_loader(self.dps[self.index])
        dp = self.dp_tfm(dp).unsqueeze(0)
        name = self.ims[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        if name.endswith('.png'):
            name = name.split('.png')[0]
        self.index += 1
        return im, gt, dp, name


def sod3d_loader(im_root, gt_root, dp_root, batch_size, train_size, shuffle=True, num_worker=12, pin_memory=True):
    dataset = SOD3DDataset(im_root, gt_root, dp_root, train_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_worker,
                                  pin_memory=pin_memory)
    return data_loader

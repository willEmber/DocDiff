import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, Resize, InterpolationMode
from PIL import Image


def ImageTransform(loadSize, resize_test: bool = False):
    test_tf = [ToTensor()]
    if resize_test:
        # Resize to configured IMAGE_SIZE for eval when not using native resolution
        test_tf = [Resize(loadSize, interpolation=InterpolationMode.BICUBIC), ToTensor()]
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ]), "test": Compose(test_tf), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1, resize_test: bool = False):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.mode = mode
        # Build a sorted intersection of filenames to ensure consistent pairing
        def list_images(p):
            return sorted([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        gt_files = set(list_images(path_gt))
        img_files = set(list_images(path_img))
        self.files = sorted(list(gt_files & img_files))
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize, resize_test=resize_test)["test"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]
        gt = Image.open(os.path.join(self.path_gt, fname))
        img = Image.open(os.path.join(self.path_img, fname))
        img = img.convert('RGB')
        gt = gt.convert('RGB')
        if self.mode == 1:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.ImgTrans[0](img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img= self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        name = fname
        return img, gt, name

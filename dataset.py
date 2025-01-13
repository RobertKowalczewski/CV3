import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as FT

from parameters import device

GTA_small_mean = torch.FloatTensor([0.5083843,  0.502183,   0.48381886]).unsqueeze(1).unsqueeze(2)
GTA_small_std = torch.FloatTensor([0.25136814, 0.24611233, 0.24406359]).unsqueeze(1).unsqueeze(2)

GTA_small_mean_cuda = torch.FloatTensor([0.5083843,  0.502183,   0.48381886]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
GTA_small_std_cuda = torch.FloatTensor([0.25136814, 0.24611233, 0.24406359]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img):
    """
    Accepts PIL Image
    """
    img = FT.to_tensor(img)
    if img.ndimension() == 3:
            img = (img - GTA_small_mean) / GTA_small_std
    elif img.ndimension() == 4:
        img = (img - GTA_small_mean_cuda) / GTA_small_std_cuda
    return img


class Transform:
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img)
        hr_img = convert_image(hr_img)

        return lr_img, hr_img

class GTA(Dataset):
    def __init__(self, img_dir, split, crop_size, scaling_factor):
        self.crop_size = crop_size
        self.split = split.lower()
        self.scaling_factor = scaling_factor
        if split == 'train':
            self.img_dir = img_dir + "x/"
        elif split == 'test':
            self.img_dir = img_dir + "y/"

        self.img_names = os.listdir(self.img_dir)
        
        self.transform = Transform(split, crop_size, scaling_factor)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path, mode='r')
        image = image.convert('RGB')
        return self.transform(image)

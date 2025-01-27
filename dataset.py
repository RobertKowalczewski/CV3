import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt

from parameters import device

GTA_small_mean = torch.FloatTensor([0.5083843,  0.502183,   0.48381886]).unsqueeze(1).unsqueeze(2)
GTA_small_std = torch.FloatTensor([0.25136814, 0.24611233, 0.24406359]).unsqueeze(1).unsqueeze(2)

GTA_small_mean_cuda = torch.FloatTensor([0.5083843,  0.502183,   0.48381886]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
GTA_small_std_cuda = torch.FloatTensor([0.25136814, 0.24611233, 0.24406359]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'gta-small-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'gta-small-norm':
        if img.ndimension() == 3:
            img = (img - GTA_small_mean) / GTA_small_std
        elif img.ndimension() == 4:
            img = (img - GTA_small_mean_cuda) / GTA_small_std_cuda

    # elif target == 'y-channel':
    #     # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
    #     # torch.dot() does not work the same way as numpy.dot()
    #     # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
    #     img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class Transform:
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor,lr_img_type, hr_img_type, same_size_input = False):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.same_size_input = same_size_input

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

        
        # If same size input and output is enabled, upscale lr_img
        if self.same_size_input:
            lr_img = lr_img.resize((int(hr_img.width), int(hr_img.height)),Image.BICUBIC)


        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img

class GTA(Dataset):
    def __init__(self, img_dir, split, crop_size, scaling_factor,lr_img_type, hr_img_type, same_size_input = False):
        self.crop_size = crop_size
        self.split = split.lower()
        self.scaling_factor = scaling_factor
        self.img_dir = img_dir
        self.img_names = os.listdir(self.img_dir)
        
        self.transform = Transform(split, crop_size, scaling_factor,lr_img_type, hr_img_type, same_size_input = same_size_input)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path, mode='r')
        image = image.convert('RGB')
        return self.transform(image)

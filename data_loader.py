import os
import glob
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import Kernels, load_kernels

TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

torch.set_default_tensor_type(torch.FloatTensor)


def Scaling(image):
    return np.array(image) / 255.0

def Scaling01(x):
    np.array(x)
    return (x - x.min())/(x.max() - x.min())

def random_downscale(image, scale_factor):
    options = {0:Image.BICUBIC, 1: Image.BILINEAR, 2: Image.NEAREST}
    downscaled_image = image.resize((np.array(image).shape[0]//scale_factor,np.array(image).shape[0]//scale_factor), options[np.random.randint(3)])
    return downscaled_image

def bicub_downscale(image, scale_factor):
    downscaled_image = image.resize((np.array(image).shape[0]//scale_factor,np.array(image).shape[0]//scale_factor), Image.BICUBIC)
    return downscaled_image

def bilin_downscale(image, scale_factor):
    downscaled_image = image.resize((np.array(image).shape[0]//scale_factor,np.array(image).shape[0]//scale_factor), Image.BILINEAR)
    return downscaled_image

def nearest_downscale(image, scale_factor):
    downscaled_image = image.resize((np.array(image).shape[0]//scale_factor,np.array(image).shape[0]//scale_factor), Image.NEAREST)
    return downscaled_image


class ImageFolder(data.Dataset):
    def __init__(self, root, config=None):
        self.device = config.device
        self.image_paths = sorted(glob.glob(root + "/*.*"))
        self.image_size = config.image_size
        self.scale_factor = config.scale_factor
        hr_height, hr_width = config.image_size * config.scale_factor, config.image_size * config.scale_factor
        K, P = load_kernels(file_path='kernels/', scale_factor=self.scale_factor)
        #K = kernels -> K.shape = (15,15,1,358)
        #P = Matriz de projeçao do PCA --> P.shape = (15,225)
        self.randkern = Kernels(K, P)
        # Transforms for low resolution images and high resolution images

    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path)
        if np.array(image).shape==4:
            image = np.array(image)[:,:,:3]
            image = Image.fromarray(image)

        # target (high-resolution image)
        
                    # Random Crop:
        #transform = transforms.RandomCrop(self.image_size * self.scale_factor)
        #hr_image = transform(image)
        
                    # Face Crop:
        face_width = face_height = 128
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        hr_image = image

        # HR_image --> [0,1] --> torch
        hr_image_scaled = Scaling(hr_image)
        hr_image_scaled = torch.from_numpy(hr_image_scaled).float().to(self.device) # NUMPY to TORCH

        # get HR residuals --> [-1,1] --> torch
        transform_HR = transforms.Compose([
                            #random blur
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)), 
                            #downscale BICUBIC pro tamanho LR
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC), 
                            #upscale BICUBIC pro tamanho HR
                            transforms.Resize((self.image_size*self.scale_factor, self.image_size*self.scale_factor), Image.BICUBIC)
        ])
        hr_image_hat = transform_HR(hr_image)
        hr_residual = np.array(hr_image).astype(float) - np.array(hr_image_hat).astype(float) 
        hr_residual_scaled = Scaling(hr_residual)
        hr_residual_scaled = torch.from_numpy(hr_residual_scaled).float().to(self.device) # NUMPY to TORCH

        # get LR_IMAGE --> [0,1] 
        transform_to_lr = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC) 
                    ])
        lr_image = transform_to_lr(hr_image)
        lr_image_scaled = Scaling(lr_image)

        # get LR_RESIDUAL --> [-1,1]
        transform_to_vlr_bicub = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: bicub_downscale(x,self.scale_factor)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        transform_to_vlr_bilin = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: bilin_downscale(x,self.scale_factor)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        transform_to_vlr_nears = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: nearest_downscale(x,self.scale_factor)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        lr_image_hat_bicub = transform_to_vlr_bicub(lr_image)
        lr_residual_bicub = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_bicub).astype(np.float32)
        lr_residual_bicub_scaled = Scaling(lr_residual_bicub)

        lr_image_hat_bilin = transform_to_vlr_bilin(lr_image)
        lr_residual_bilin = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_bilin).astype(np.float32)
        lr_residual_bilin_scaled = Scaling(lr_residual_bilin)

        lr_image_hat_nears = transform_to_vlr_nears(lr_image)
        lr_residual_nears = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_nears).astype(np.float32)
        lr_residual_nears_scaled = Scaling(lr_residual_nears)

        # LR_image_scaled + kernel_proj + LR_residual_scaled (CONCAT) ---> TO TORCH

        lr_image_without_kernel = lr_image_scaled #self.randkern.ConcatDegraInfo(lr_image_scaled)
        lr_image_with_resid  = np.concatenate((lr_image_without_kernel,\
                                               lr_residual_bicub_scaled,\
                                               lr_residual_bilin_scaled,\
                                               lr_residual_nears_scaled), axis=-1)
        lr_image_with_resid = torch.from_numpy(lr_image_with_resid).float().to(self.device) # NUMPY to TORCH

        # LR_image to torch
        lr_image_scaled = torch.from_numpy(lr_image_scaled).float().to(self.device) # NUMPY to TORCH

        #Transpose - Permute since for model we need input with channels first
        lr_image_scaled = lr_image_scaled.permute(2,0,1) 
        hr_image_scaled = hr_image_scaled.permute(2,0,1) 
        lr_image_with_resid = lr_image_with_resid.permute(2,0,1)
        hr_residual_scaled = hr_residual_scaled.permute(2,0,1)
        
        return image_path,\
                lr_image_scaled,\
                hr_image_scaled, \
                lr_image_with_resid, \
                hr_residual_scaled

    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)


def get_loader(image_path, config):
    """Create and return Dataloader."""
    dataset = ImageFolder(image_path, config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    return data_loader

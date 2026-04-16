#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from typing import List
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import torch
import torchvision
import OpenEXR
import Imath
import numpy as np
import cv2
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Example: save_exr(tensor, "output.exr")
def save_exr(tensor, filename):
    # Ensure the tensor is on the CPU and convert to numpy array
    data = tensor.cpu().numpy().astype(np.float32)

    # Define the EXR header
    channel_type = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header = OpenEXR.Header(tensor.shape[2], tensor.shape[1])

    # Check if the tensor has 3 channels (RGB)
    if tensor.shape[0] == 3:
        header['channels'] = {
            'R': channel_type,
            'G': channel_type,
            'B': channel_type }
    elif tensor.shape[0] == 1:
        header['channels'] = {'Y': channel_type}
    else:
        raise ValueError("Tensor must have 1 or 3 channels (RGB)")

    exr_file = OpenEXR.OutputFile(filename, header)

    if tensor.shape[0] == 1:
        exr_file.writePixels({'Y': data[0,:,:].tobytes()})
    else:
        exr_file.writePixels({
            'R': data[0,:,:].tobytes(),
            'G': data[1,:,:].tobytes(),
            'B': data[2,:,:].tobytes()})

    exr_file.close()

def read_exr(filename, device = "cuda") -> torch.Tensor:
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to extract the image size
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel type
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if len(header['channels']) == 1:
        # Read the Y channel
        y_channel = np.frombuffer(exr_file.channel('Y', pt), dtype=np.float32).reshape(height, width)
        image = np.stack([y_channel], axis=0)
    elif len(header['channels']) == 3:
        # Read the RGB channels
        r_channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(height, width)
        g_channel = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(height, width)
        b_channel = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(height, width)
        image = np.stack([r_channel, g_channel, b_channel], axis=0)
    else:
        raise ValueError("EXR file must have 1 or 3 channels")

    # Convert the numpy array to a PyTorch tensor
    image = torch.from_numpy(image)
    if device == "cuda":
        image = image.cuda()
    elif device != "cpu":
        raise ValueError("Device must be 'cuda' or 'cpu'")
    return image


def save_images_multithread(tensors: List[torch.Tensor],
                            paths: List[str],
                            workers: int = 4):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for tensor, path in zip(tensors, paths):
            executor.submit(torchvision.utils.save_image, tensor, path)


# Example: load_images_multithread("/dataset", 8)
def load_images_multithread(directory: str, workers: int = 4) -> List[torch.Tensor]:
    
    def load_image(path, transform):
        return transform(Image.open(path)).cuda()

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    paths = [f"{directory}/{file}" for file in os.listdir(directory)]
    images = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        transform = torchvision.transforms.ToTensor()
        for path in paths:
            futures.append(executor.submit(load_image, path, transform))

        for future in futures:
            images.append(future.result())

    return images


def max_k_points(texture:torch.Tensor, k, gap = 0.01):
    C, H, W = texture.shape
    assert C == 1

    min_dist2 = min(H, W) * gap
    min_dist2 = min_dist2 * min_dist2
    sort_length = min(256 * k, H * W)
    
    rst = torch.zeros((k, 2), dtype=torch.int).cuda()
    count = 0

    buffer_ptr = 1
    while count < k:
        _, idx = torch.topk(texture.view(1,-1), sort_length)
        coord_buffer = torch.cat([idx // W, idx % W], dim=0).t()

        if count == 0:
            rst[0] = coord_buffer[0]
            count = 1
        
        while (buffer_ptr < sort_length) and (count < k):
            dist2 = torch.sum(
                (rst[:count] - coord_buffer[buffer_ptr:buffer_ptr+1])**2, dim=1).min()
            if dist2 > min_dist2:
                rst[count] = coord_buffer[buffer_ptr]
                count += 1
            buffer_ptr += 1
        
        sort_length = min(2 * sort_length, H * W)

    return rst

def tensors_to_mp4(tensor_list, output_path, fps=30, progress=True):
    """
        tensor_list: List of tensors of shape (3, H, W)
    """
    if not tensor_list:
        raise ValueError("tensor list cannot be empty")
    
    _, H, W = tensor_list[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"cannot open video writer with path: {output_path}")
    
    iterator = tqdm(tensor_list, desc="Exporting video") if progress else tensor_list
    for tensor in iterator:
        tensor = tensor.cpu()
        
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            tensor = (tensor * 255).clamp(0, 255)
        tensor = tensor.to(torch.uint8)
        
        frame = tensor.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        video_writer.write(frame)
    
    video_writer.release()
    print(f"video export completed: {output_path}")


def srgb_to_linear(image_tensor):
    """
        image_tensor: (C, H, W) / (B, C, H, W)
    """

    if image_tensor.min() < 0 or image_tensor.max() > 1:
        raise ValueError("Input tensor values must be in the range [0, 1]")
    
    mask = image_tensor <= 0.04045
    linear_tensor = torch.where(
        mask,
        image_tensor / 12.92,
        ((image_tensor + 0.055) / 1.055) ** 2.4
    )
    
    return linear_tensor

def linear_to_srgb(linear_tensor):
    """
        linear_tensor: (C, H, W) / (B, C, H, W)
    """
    mask = linear_tensor <= 0.0031308
    srgb_tensor = torch.where(
        mask,
        linear_tensor * 12.92,
        1.055 * (linear_tensor ** (1.0 / 2.4)) - 0.055
    )
    return torch.clamp(srgb_tensor, 0.0, 1.0)
import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.gridspec as gridspec

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision.io import read_image
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

import torchvision.transforms as transforms


def imsc(img, *args, quiet=False, lim=None, interpolation='lanczos', **kwargs):
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    handle = None
    with torch.no_grad():
        if not lim:
            lim = [img.min(), img.max()]
        img = img - lim[0]  # also makes a copy
        img.mul_(1 / (lim[1] - lim[0]))
        img = torch.clamp(img, min=0, max=1)
        if not quiet:
            bitmap = img.expand(3,
                                *img.shape[1:]).permute(1, 2, 0).cpu().numpy()
    return bitmap

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def imagenet_image_transforms(device: str, new_shape_of_image: int = 224):
    """
    Returns transformations that takes a torch tensor and transforms it into a new tensor
    of size (1, C, new_shape_of_image, new_shape_of_image), normalizes the image according
    to the statistics from the Imagenet dataset, and puts the tensor on the desired device.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_shape_of_image, new_shape_of_image)),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        torchvision.transforms.Lambda(unsqeeze_image),
        ToDevice(device),
    ])

    return transform


# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0).to('cuda')
])


read_img = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    lambda x: x.to('cuda')
])





class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"

def unsqeeze_image(input_image: torch.Tensor) -> torch.Tensor:
    return input_image.unsqueeze(0)





grayscale_trans = transforms.Grayscale()


def grayscale(img):
    img = grayscale_trans(img)
    img = img.repeat(1, 3, 1, 1)
    return img


#blur = transforms.GaussianBlur((5, 5), sigma=(3, 4.0))
blur = transforms.GaussianBlur((5, 5), sigma=(3, 4.0))


class Net(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(Net, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                                    padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                                  padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                               padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0],
                             [0, 1, -1],
                             [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0],
                               [-1, 1, 0],
                               [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1],
                               [0, 1, 0],
                               [0, 0, 0]])

        all_filters = np.stack(
            [filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape,
                                            padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
        grad_mag += torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2)
        grad_mag += torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2)
        grad_orientation = (
                    torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (180.0 / 3.14159))
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1, height, width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1, height, width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges < self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag < self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()
        # blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold
        edges=thin_edges.repeat(1, 3, 1, 1)
        edges = (edges.max() - edges) / edges.max()
        return edges, early_threshold.repeat(1, 3, 1, 1)


use_cuda = True

edge_detect = Net(threshold=3.0, use_cuda=use_cuda)
if use_cuda:
    edge_detect.cuda()
edge_detect.eval()



def mask_generator(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 7,
    probablity_of_drop: float = 0.5,
    num_spatial_dims: int = 2) -> torch.Tensor:
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using bilinear interpolation to obtain smooth continious mask in (0, 1).
    """

    pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

    grid = (torch.rand(batch_size, 1, *((num_cells,) * num_spatial_dims), device=device) < probablity_of_drop).float()
    grid_up = F.interpolate(grid, size=(shape), mode='bilinear', align_corners=False)
    grid_up = F.pad(grid_up, pad_size, mode='reflect')

    shift_x = torch.randint(0, num_cells, (batch_size,), device=device)
    shift_y = torch.randint(0, num_cells, (batch_size,), device=device)

    masks = torch.empty((batch_size, 1, shape[0], shape[1]), device=device)

    for mask_i in range(batch_size):
        masks[mask_i] = grid_up[
            mask_i,
            :,
            shift_x[mask_i]:shift_x[mask_i] + shape[0],
            shift_y[mask_i]:shift_y[mask_i] + shape[1]
        ]

    yield masks
'''
def mask_generator_binary(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 8,
    probablity_of_drop: float = 0.1,
    num_spatial_dims: int = 2) -> torch.Tensor:
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using bilinear interpolation to obtain smooth continious mask in (0, 1).
    """

    pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

    grid = (torch.rand(batch_size, 1, *((num_cells,) * num_spatial_dims), device=device) < probablity_of_drop).float()
    grid_up = F.interpolate(grid, size=(shape), mode='bilinear', align_corners=False)
    grid_up = F.pad(grid_up, pad_size, mode='reflect')

    shift_x = torch.randint(0, num_cells, (batch_size,), device=device)
    shift_y = torch.randint(0, num_cells, (batch_size,), device=device)

    masks = torch.empty((batch_size, 1, shape[0], shape[1]), device=device)

    for mask_i in range(batch_size):
        masks[mask_i] = grid_up[
            mask_i,
            :,
            shift_x[mask_i]:shift_x[mask_i] + shape[0],
            shift_y[mask_i]:shift_y[mask_i] + shape[1]
        ]
    masks=(masks>0.5).float()

    yield masks




import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from tqdm import tqdm

def mask_generator_rise(
    N: int,
    shape: tuple,
    device: str,
    s: int = 8,
    p1: float = 0.1) -> torch.Tensor:

    cell_size = torch.ceil(torch.tensor(shape, dtype=torch.float32) / s)
    up_size = ((s + 1) * cell_size).long()

    grid = torch.rand(N, s, s) < p1
    grid = grid.float()

    masks = torch.empty((N, *shape))

    for i in range(N):
        # Random shifts
        x = torch.randint(0, int(cell_size[0]), (1,)).item()
        y = torch.randint(0, int(cell_size[1]), (1,)).item()
        # Linear upsampling and cropping
        # Resizing needs to be adjusted to use torchvision or custom implementation
        resized = F.interpolate(grid[i], size=up_size, mode='bilinear', align_corners=True)
        masks[i, :, :] = resized[:, :, x:x + shape[0], y:y + shape[1]].squeeze()
    masks = masks.unsqueeze(1)
    masks = masks.cuda()  # Assuming you want to move the masks to GPU
    yield masks

'''




import torchvision.transforms as transforms



def binary_mask_generator(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 14,
    probablity_of_drop: float = 0.7,
    num_spatial_dims: int = 2) -> torch.Tensor:
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using bilinear interpolation to obtain smooth continious mask in (0, 1).
    """
    crop = transforms.RandomResizedCrop(size=shape,scale=(0.1, 0.5))

    pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

    grid = (torch.rand(batch_size, 1, *((num_cells,) * num_spatial_dims), device=device) < probablity_of_drop).float()
    grid_up = F.interpolate(grid, size=(224*16,224*16), mode='bicubic', align_corners=True)
    masks=crop(grid_up)
    yield masks
    
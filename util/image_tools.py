import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

transform = transforms.ToTensor()


def convert_image_to_tensor(image):
    return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0, 2, 3, 1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0, 2, 3, 1))
    else:
        raise Exception("This tensor must have 4 dimension.")

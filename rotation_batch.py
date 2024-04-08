# REF: https://kornia.github.io/tutorials/nbs/rotate_affine.html#define-the-rotation-matrix
import cv2
import kornia as K
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

# S1: load the image using kornia
x_img = K.io.load_image("/home/huynn/huynn_ws/edge-server-project/esim_ros_ws/src/kornia_tutorial/car.png", K.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW
print('type(x_img) ', type(x_img))
def imshow(input: torch.Tensor, size: tuple = None):
    out = torchvision.utils.make_grid(input, nrow=4, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=size)
    plt.imshow(out_np)
    plt.axis("off")
    plt.show()


x_batch = x_img.repeat(16, 1, 1, 1)
x_rot = K.geometry.rotate(x_batch, torch.linspace(0.0, 360.0, 16))

imshow(x_rot, (16, 16))
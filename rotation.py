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


# S2: Define the rotation matrix
#   - create transformation (rotation)
alpha: float = 45.0  # in degrees
angle: torch.tensor = torch.ones(1) * alpha

#   - define the rotation center
center: torch.tensor = torch.ones(1, 2)     # 1 dòng và 2 cột -> vector hàng
center[..., 0] = x_img.shape[3] / 2  # x
center[..., 1] = x_img.shape[2] / 2  # y

#   - define the scale factor
scale: torch.tensor = torch.ones(1, 2)

#   - compute the transformation matrix
conversion_matrix: torch.tensor = K.geometry.get_rotation_matrix2d(center, angle, scale)  # 1x2x3

# S3: Apply the transformation to the original image
_, _, h, w = x_img.shape
x_warped: torch.tensor = K.geometry.warp_affine(x_img, conversion_matrix, dsize=(h, w))

imshow(x_warped)
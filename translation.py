# REF: https://kornia.github.io/tutorials/nbs/rotate_affine.html#define-the-rotation-matrix
import cv2
import kornia as K
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

# S1: load the image using kornia
x_img = K.io.load_image("/home/huynn/huynn_ws/edge-server-project/esim_ros_ws/src/kornia_tutorial/car.png", K.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW

def imshow(input: torch.Tensor, size: tuple = None):
    out = torchvision.utils.make_grid(input, nrow=4, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.figure(figsize=size)
    plt.imshow(out_np)
    plt.axis("off")
    plt.show()


# S2: Define the rotation matrix
#   - create transformation (translation)
#   - Đặt lượng dịch chuyển bạn muốn áp dụng
dx, dy = 10, 20  # Dịch chuyển 10 pixels theo x và 20 pixels theo y

#   - Tạo ma trận dịch chuyển affine
conversion_matrix = torch.tensor([[1, 0, dx], 
                                   [0, 1, dy]])

#   - Bởi vì Kornia yêu cầu tensor đầu vào là float, chúng ta cần chuyển đổi kiểu dữ liệu
conversion_matrix = conversion_matrix.float()
print('conversion_matrix.shape: ', conversion_matrix.shape)

#   - Thêm một chiều batch_size và channels nếu cần (giả sử img là ảnh đơn không có batch và single channel)
conversion_matrix = conversion_matrix[None, ...]
# x_img = x_img[None, None, ...]

# S3: Apply the transformation to the original image
_, _, h, w = x_img.shape

x_warped: torch.tensor = K.geometry.warp_affine(x_img, conversion_matrix, dsize=(h, w))

imshow(x_warped)
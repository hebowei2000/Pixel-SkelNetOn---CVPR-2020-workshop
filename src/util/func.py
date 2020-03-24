import numpy as np

import torch
import torch.nn.functional as F


def label_edge_prediction(label):
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = torch.from_numpy(fx).cuda()
    fy = torch.from_numpy(fy).cuda()
    contour_th = 1.5
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, [1, 1, 1, 1], mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

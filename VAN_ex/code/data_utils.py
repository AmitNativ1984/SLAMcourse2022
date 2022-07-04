import cv2
import os
import numpy as np
from params import *

def read_image_pair(data_path, idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(data_path, "image_0", img_name),0)
    img2 = cv2.imread(os.path.join(data_path, "image_1", img_name),0)
    return img1, img2

def read_cameras(data_path):
    with open(os.path.join(data_path,'calib.txt')) as f:
        l1 = f.readline().split()[1:]       # skip first token
        l2 = f.readline().split()[1:]       # skip first token

    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2
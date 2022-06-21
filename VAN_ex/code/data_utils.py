import cv2
import os
def read_image_pair(data_path, idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(data_path, "image_0", img_name),0)
    img2 = cv2.imread(os.path.join(data_path, "image_1", img_name),0)
    return img1, img2
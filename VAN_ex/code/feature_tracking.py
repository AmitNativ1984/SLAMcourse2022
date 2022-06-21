import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_detect_and_compute(img1):
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    return kp1, des1


def match_descriptors(des1, des2):
    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    return matches

def sort_matches(matches):
    # Sort matches by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)
    return matches
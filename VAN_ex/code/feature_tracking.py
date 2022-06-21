import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_detect_and_compute(img1):
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    return kp1, des1


def match_descriptors_knn(des1, des2, k=2):
    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k)
    return matches

def filter_matches_by_significance_test(matches, ratio_th=0.7):
    # Discard matches with segnificance test
    good_matches = []
    bad_matches = []
    for m, n in matches:
        if m.distance < ratio_th * n.distance:
            good_matches.append(m)
        else:
            bad_matches.append(m)
    return good_matches, bad_matches

def sort_matches_by_distance(matches):
    # Sort matches by their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return matches
import cv2
import numpy as np
from data_utils import *
from feature_tracking import *
import random

if __name__ =='__main__':
    DATA_PATH = "../../VAN_ex/dataset/sequences/00"
    # Read image pair
    img1, img2 = read_image_pair(DATA_PATH, 0)
    
    # Detect ORB features and compute descriptors
    kp1, des1 = orb_detect_and_compute(img1)
    kp2, des2 = orb_detect_and_compute(img2)

    # print descriptors of first two features:
    print("Descriptors of first two features:")
    print(des1[:2])

    # Match descriptors
    matches = match_descriptors_knn(des1, des2, k=2)
    
    ####################################
    # select random indexes of matches
    ####################################
    num_selections = 20
    print("randomly selecting {} matches".format(num_selections))
    random_20_matches = [m[0] for m in random.sample(matches, num_selections)]
    random_matches_img = cv2.drawMatches(img1, kp1, img2, kp2, random_20_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure("randomly selected matches")
    plt.imshow(random_matches_img)
    
    ########################################
    # discard matches with segnificance test
    ########################################
    # matches used knn and return 2 best matches for every feature point.
    good_matches, bad_matches = filter_matches_by_significance_test(matches, ratio_th=0.7)
    
    print("Number of good matches: {}".format(len(good_matches)))
    print("Number of discarded matches: {}".format(len(matches) - len(good_matches)))
    significance_test_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure("significance test")
    plt.imshow(significance_test_img)

    # correct match that failed the significance test
    bad_matches = sort_matches_by_distance(bad_matches)
    significance_test_false_img = cv2.drawMatches(img1, kp1, img2, kp2, bad_matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure("significance test false failure")
    plt.imshow(significance_test_false_img)

    plt.show()

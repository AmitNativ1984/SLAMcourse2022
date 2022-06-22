import cv2
import numpy as np
from data_utils import *
from feature_tracking import *
import random

if __name__ == '__main__':
    print("\n")
    print("*** Start ex2 ***")
    DATA_PATH = "../../VAN_ex/dataset/sequences/00"
    # Read image pair
    img1, img2 = read_image_pair(DATA_PATH, 0)

    # Detect ORB features and compute descriptors
    kp1, des1 = orb_detect_and_compute(img1)
    kp2, des2 = orb_detect_and_compute(img2)

    ##################################################################
    # We are working with rectified images. So, all matches should lie
    # on the same epipolar line.
    # The reason for the pattern is that one camera has been 
    # transformed to be parallel to the other.
    ##################################################################

    # match descriptors
    matches = [m[0] for m in match_descriptors_knn(des1, des2, k=2)]

    # calculate pixel deviations from epipolar line
    dist = matches_deviation_from_rectified_stereo_pattern(kp1, kp2, matches)
    
    # create histogram of deviations
    hist, bins = np.histogram(dist, bins=list(range(0, 250, 1)))
    plt.figure("histogram of deviations")
    plt.bar(bins[:-1], hist, width=1)
    plt.xlabel("deviation from rectified stereo pattern")
    plt.ylabel("number of matches")


    # max allowed deviation from rectified stereo pattern
    max_rect_deviation = 2
    print("Percentage of matches that deviate from rectified stereo pattern by more than {} pixels: {}%".format(max_rect_deviation, sum(hist[max_rect_deviation:])/sum(hist)*100))

    # filter matches by deviation from rectified stereo pattern:
    good_matches, bad_matches = filter_matches_by_rectified_stereo_pattern_constraint(kp1, kp2, matches, max_rect_deviation=max_rect_deviation)

    # plot matches on image pair:
    matches_img1 = cv2.drawKeypoints(img1, [kp1[m.queryIdx] for m in good_matches], None, color=[255, 165, 0], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_img1 = cv2.drawKeypoints(matches_img1, [kp1[m.queryIdx] for m in bad_matches], None, color=[0, 255, 255], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matches_img2 = cv2.drawKeypoints(img2, [kp2[m.trainIdx] for m in good_matches], None, color=[255, 165, 0], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_img2 = cv2.drawKeypoints(matches_img2, [kp2[m.trainIdx] for m in bad_matches], None, color=[0, 255, 255], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    rectified_img_matches = np.vstack((matches_img1, matches_img2))

    plt.figure("matches by rectified stereo pattern")
    plt.imshow(rectified_img_matches)
    
    # Assuming Y values of erroneous matches are distributed uniformly accross the image, the ratio of
    # matches that are still wrong after this rejection policy, is
    des1_good_matches = np.asarray([des1[m.queryIdx] for m in good_matches])
    des2_good_matches = np.asarray([des2[m.trainIdx] for m in good_matches])
    
    matches_knn = match_descriptors_knn(des1_good_matches, des2_good_matches, k=2)
    _, good_matches_below_significance_level = filter_matches_by_significance_test(matches_knn, ratio_th=0.7)
   
    print("Probability of match that passes rectified contrained, to be erroneous: {}%".format(len(good_matches_below_significance_level)/len(good_matches)*100))
    print("Total probability of match to pass rectified stereo pair constraint and fail significance test: {}%".format(len(good_matches_below_significance_level)/len(good_matches) * len(good_matches)/len(matches) * 100))
    plt.show()
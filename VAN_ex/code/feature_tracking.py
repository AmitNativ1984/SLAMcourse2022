import cv2
import numpy as np
import matplotlib.pyplot as plt
from params import *

def orb_detect_and_compute(img1):
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    return kp1, des1


def match_descriptors_knn(des1, des2, k=2):
    # des1: descriptors of query image (the image where we look for matching features)
    # des2: descriptors of train image (the image we to extract features that need to be found in query image)
    
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

def matches_deviation_from_rectified_stereo_pattern(kp1, kp2, matches):
    # Assumption: kp1 and kp2 are matches of rectified image pair.
    # Compute the deviation of the matches from the epipolar line.
    # kp1: keypoints of query image
    # kp2: keypoints of train image
    # matches: matches between query image and train image

    # euclidian distance between keypoints:
    dist = [np.linalg.norm(np.asarray(kp1[m.queryIdx].pt[1]) - np.asarray(kp2[m.trainIdx].pt[1])) for m in matches]
    return dist

def filter_matches_by_rectified_stereo_pattern_constraint(kp1, kp2, matches, max_rect_deviation=5):
    # Assumption: kp1 and kp2 are matches of rectified image pair.
    # Filter matches that deviate from the rectified stereo pattern.
    # kp1: keypoints of query image
    # kp2: keypoints of train image
    # matches: matches between query image and train image
    # max_rect_deviation: maximum allowed deviation from rectified stereo pattern
    # return: filtered matches

    # calculate pixel deviations from epipolar line
    dist = matches_deviation_from_rectified_stereo_pattern(kp1, kp2, matches)
    # filter matches that deviate from rectified stereo pattern
    good_matches = []
    bad_matches = []
    for m, d in zip(matches, dist):
        if d <= max_rect_deviation:
            good_matches.append(m)
        else:
            bad_matches.append(m)
    
    return good_matches, bad_matches

def get_consistent_matches_between_successive_frames(kp, good_matches_between_frames, matches_between_pairs):
    # filter keypoints and matches that match between successive frames and are consistent with image pairs.
    # Returns: [(match_pair0_0, match_pair1_0), (match_pair0_1, match_pair1_1), (match_pair0_2, match_pair1_2), ...]
    #           A list of tuples representing the same point in all 4 images
    
    consistent_matches = []
    kp_img_pair_0_left = [kp[FRAME0][LEFT][match.queryIdx] for match in matches_between_pairs[FRAME0]]
    kp_img_pair_1_left = [kp[FRAME1][LEFT][match.queryIdx] for match in matches_between_pairs[FRAME1]]

    for match in good_matches_between_frames:
        # get keypoint of left FRAME0:
        kp_frame0 = kp[FRAME0][LEFT][match.queryIdx]    #return pointer to keypoint
        # get matching keypoint of left FRAME1:
        kp_frame1 = kp[FRAME1][LEFT][match.trainIdx]    #return pointer to keypoint

        # check if keypoints are consistent with image pairs:
        try:
            left_match = kp_img_pair_0_left.index(kp_frame0)
            right_match = kp_img_pair_1_left.index(kp_frame1)
            consistent_matches.append((matches_between_pairs[FRAME0][left_match], matches_between_pairs[FRAME1][right_match]))
        except Exception:
            continue

    return consistent_matches

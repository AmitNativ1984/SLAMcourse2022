import cv2
import numpy as np
import matplotlib.pyplot as plt
from params import *
import logging
logger = logging.getLogger(__name__)

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

def consistent_matches_between_successive_frames(kp, good_matches_between_frames, matches_between_pairs):
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
            left_match_0 = kp_img_pair_0_left.index(kp_frame0)
            left_match_1 = kp_img_pair_1_left.index(kp_frame1)
            consistent_matches.append((matches_between_pairs[FRAME0][left_match_0], matches_between_pairs[FRAME1][left_match_1]))
        except Exception:
            continue

    return consistent_matches

def get_transformation_supporters(K, P, Q, T, kp, matches, point_cloud, px_dist=2):
    # return all matches that are supporters of the transformation T
    # T: the tranformation of the object as seen in the camera coordinate system
    # kp: keypoints [(kp_frame0_left, kp_frame0_right), (kp_frame1_left, kp_frame1_right)]
    # matches: matches [(matches_frame0, matches_frame1)]
    # point_cloud: point cloud
    # px_dist: maximum distance between projected point to keypoint

    uv_proj = [[[], []], [[], []]]
    X = np.vstack((point_cloud, np.ones((1, point_cloud.shape[1]))))
    P = np.vstack((P, np.array([0, 0, 0, 1])))
    Q = np.vstack((Q, np.array([0, 0, 0, 1])))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    # dimensionality reduction
    M = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]])
    # frame0: left image
    uv_proj[FRAME0][LEFT] = K @ M @ P @ X
    # frame0: right image
    uv_proj[FRAME0][RIGHT] = K @ M @ Q @ X
    # frame1: left image
    uv_proj[FRAME1][LEFT] = K @ M @ P @ T @ X
    # frame1: right image
    uv_proj[FRAME1][RIGHT] = K @ M @ Q @ T @ X

    # normalizing uv_proj in homogeneous coordinates to get pixels:
    uv_proj[FRAME0][LEFT] = (uv_proj[FRAME0][LEFT][:2, :] / uv_proj[FRAME0][LEFT][2, :]).transpose()
    uv_proj[FRAME0][RIGHT] = (uv_proj[FRAME0][RIGHT][:2, :] / uv_proj[FRAME0][RIGHT][2, :]).transpose()
    uv_proj[FRAME1][LEFT] = (uv_proj[FRAME1][LEFT][:2, :] / uv_proj[FRAME1][LEFT][2, :]).transpose()
    uv_proj[FRAME1][RIGHT] = (uv_proj[FRAME1][RIGHT][:2, :] / uv_proj[FRAME1][RIGHT][2, :]).transpose()
    
    uv = [[[], []], [[], []]]
    for match in matches:
        uv[FRAME0][LEFT].append(kp[FRAME0][LEFT][match[FRAME0].queryIdx].pt)
        uv[FRAME0][RIGHT].append(kp[FRAME0][RIGHT][match[FRAME0].trainIdx].pt)

        uv[FRAME1][LEFT].append(kp[FRAME1][LEFT][match[FRAME1].queryIdx].pt)
        uv[FRAME1][RIGHT].append(kp[FRAME1][RIGHT][match[FRAME1].trainIdx].pt)

    # calculate distance:
    supporters = []
    for idx in range(len(uv[FRAME0][LEFT])):
        d1 = np.linalg.norm(uv_proj[FRAME0][LEFT][idx] - uv[FRAME0][LEFT][idx])
        d2 = np.linalg.norm(uv_proj[FRAME0][RIGHT][idx] - uv[FRAME0][RIGHT][idx])
        d3 = np.linalg.norm(uv_proj[FRAME1][LEFT][idx] - uv[FRAME1][LEFT][idx])
        d4 = np.linalg.norm(uv_proj[FRAME1][RIGHT][idx] - uv[FRAME1][RIGHT][idx])
        if d1 <= px_dist and d2 <= px_dist and d3 <= px_dist and d4 <= px_dist:
            supporters.append(idx)

    
    return supporters

def Ransac(K, P, Q, kp, matches, point_clouds, RANSAC_iterations=1000, RANSAC_thershold=2, num_random_choices=4):
    # returns PnP tranform with RANSAC as inner kernel.
    # the ouput transform describes world coordinates as seen in camera 2 coordinate system
    # To get camera 2 pos the result need to be [R'|-R't]
    max_num_supporters = 0
    best_supporters_idx = []
    for i in range(RANSAC_iterations):
        # randomlly select 4 keypoints from the good matches
        random_indices = np.random.choice(len(matches), num_random_choices, replace=False)
        
        # calculate PnP:
        success, r1, t1 = cv2.solvePnP(objectPoints=np.array([point_clouds[FRAME0][:,i] for i in random_indices]),
                                    imagePoints=np.array([kp[FRAME1][LEFT][matches[:, FRAME1][i].queryIdx].pt for i in random_indices]),
                                    cameraMatrix=K,
                                    distCoeffs=np.zeros((4,1)),
                                    flags=cv2.SOLVEPNP_P3P)

        if not success:
            continue
        ####################################################
        # Get supporters of the transform found with PnP
        ####################################################
        R, _ = cv2.Rodrigues(r1)
        T = np.hstack((R, t1))
        supporters_idx = get_transformation_supporters(K, P, Q, T, kp, matches, point_clouds[FRAME0], px_dist=RANSAC_thershold)
        if len(supporters_idx) > max_num_supporters:
            logger.info("found new best PnP transformation with {} supporters".format(len(supporters_idx)))
            max_num_supporters = len(supporters_idx)
            best_supporters_idx = supporters_idx
            best_T = T
            best_r1 = r1
            best_t1 = t1

        # breaking to increase iteration speed
        if len(supporters_idx) >= len(matches) * 0.5 and len(supporters_idx) > 6:
            break

    # Refine the transformation with the best supporters
    # This is done with iterative PnP on the supporters with thre previous transformation as initial guess
    # It actually performs DLT with the supporters to find the transformation
    if len(best_supporters_idx) < 6:
        raise ValueError("Not enough supporters found: {}".format(len(best_supporters_idx)))
    
    success, r1, t1 = cv2.solvePnP(objectPoints=np.array([point_clouds[FRAME0][:,i] for i in best_supporters_idx]),
                                    imagePoints=np.array([kp[FRAME1][LEFT][matches[:, FRAME1][i].queryIdx].pt for i in best_supporters_idx]),
                                    cameraMatrix=K,
                                    distCoeffs=np.zeros((4,1)),
                                    rvec= best_r1,
                                    tvec= best_t1,
                                    flags=cv2.SOLVEPNP_ITERATIVE,
                                    useExtrinsicGuess=True)
                                
                                
    R, _ = cv2.Rodrigues(r1)
    T = np.hstack((R, t1))
    return T, supporters_idx

def get_camera_pose(T):
    # T: projection matrix from PnP (tranforms world coordinates to camera coordinates)
    # returns the camera rotation and position in world coordinates: [R'|-R't]
    
    R = T[:3, :3]
    t = (T[:3, 3]).reshape(-1,1)

    return np.hstack((R.transpose(), -R.transpose()@t))
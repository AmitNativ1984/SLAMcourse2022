from cProfile import label
import cv2
import numpy as np
from data_utils import *
from feature_tracking import *
from triangulation import *
from params import *

if __name__ == '__main__':
    print("\n")
    print("*** Start ex3 ***")
    K, P, Q = read_cameras(DATA_PATH)
    des = []
    kp = []
    matches_between_pairs = []
    point_clouds = []
    max_pix_deviation = 2 #  from rectified stereo pattern
    for i in range(2):
        print("* Processing image {}".format(i))
        img1, img2 = read_image_pair(DATA_PATH, i)
        kp1, des1 = orb_detect_and_compute(img1)
        kp2, des2 = orb_detect_and_compute(img2)
        # match descriptors
        curent_pair_matches = [m[0] for m in match_descriptors_knn(des1, des2, k=2)]
        good_matches, bad_matches = filter_matches_by_rectified_stereo_pattern_constraint(kp1, kp2, curent_pair_matches, max_rect_deviation=max_pix_deviation)
        X = linear_least_squares_triangulation(K@P, kp1, K@Q, kp2, good_matches)

        kp.append((kp1, kp2))
        des.append((des1, des2))
        matches_between_pairs.append((good_matches))
        point_clouds.append((X))
        print("\tFinished processing key points, descriptors and matches for image {}".format(i))

        fig = plt.figure("3D points pair {}".format(i))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='o', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    print("\n")
    print("* Matching key points of successive left images")
    matches_frame0_frame1 = match_descriptors_knn(des[FRAME0][LEFT], des[FRAME1][LEFT], k=2)
    good_matches_between_frames, bad_matches_between_frames = filter_matches_by_significance_test(matches_frame0_frame1, ratio_th=0.7)

    # filter keypoints that are matched between left-right pairs and between frames:
    good_matches = get_consistent_matches_between_successive_frames(kp, 
                                                                    good_matches_between_frames,
                                                                    matches_between_pairs)

    good_matches = np.array(good_matches)
    print("\tFinished matching key points of successive left images")
    print("\tWe now have matches of same point between 2 successive stereo pairs")
    print("\n")

    ############################################################
    #  PNP of 4 keypoints that were matched on all four images #
    ############################################################
    print("* PNP of 4 keypoints that were matched on all four images")
    point_clouds = []
    X = linear_least_squares_triangulation(K@P, kp[FRAME0][LEFT], K@Q, kp[FRAME0][RIGHT], good_matches[:,FRAME0]).transpose()
    point_clouds.append(X[:3, :])

    # randomlly select 4 keypoints from the good matches
    random_indices = np.random.choice(len(good_matches), 4, replace=False)
    
    
    # calculate PnP:
    success, r1, t1 = cv2.solvePnP(objectPoints=np.array([point_clouds[FRAME0][:,i] for i in random_indices]),
                                   imagePoints=np.array([kp[FRAME1][LEFT][good_matches[:, FRAME1][i].queryIdx].pt for i in random_indices]),
                                   cameraMatrix=K,
                                   distCoeffs=np.zeros((4,1)),
                                   flags=cv2.SOLVEPNP_P3P)

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   * [R|t] describes the world coordinate system as seen from left1 coordinate system. left1 = R(left0) + t
    #   * The chained transform from camera A to camera B to camera C:
    #     Ta->b = R1x+t1
    #     Tb->c = R2x+t2 = R2(R1x+t1)+t2 = R2R1x+R2t1+t2 = [R2R1|R2t1+t2]
    #   * The rotation and translation of left1 camera in world coordinates (its position), is given by:
    #     R1 = [R'|-R't]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # plotting camera positions in world coordinates:
    pos0_left = np.array([[0, 0, 0]])
    pos0_right = pos0_left - Q[:,-1]

    pos1_left = pos0_left + t1.transpose()
    pos1_right = pos0_right + t1.transpose()

    pos_left = np.vstack((pos0_left, pos1_left))
    pos_right = np.vstack((pos0_right, pos1_right))

    plt.figure('camera positions in world coordinates')
    plt.scatter(pos_left[:, 0], pos_left[:, -1], color='r', marker='x', alpha=0.5, label='left camera')
    plt.scatter(pos_right[:, 0], pos_right[:, -1], color='b', marker='o', alpha=0.5, label='right camera')
    plt.legend()

plt.show()
    
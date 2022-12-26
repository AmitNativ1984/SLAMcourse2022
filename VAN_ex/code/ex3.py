from cProfile import label
import cv2
import numpy as np
from data_utils import *
from tracking.feature_tracking import *
from triangulation import *
from params import *
import time
import logging

logging.basicConfig(format='%(asctime)s [%(name)s] [%(funcName)s] [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S ', level=logging.INFO)
logger = logging.getLogger(__name__)

def first_two_frames():
    K, P, Q = read_cameras(DATA_PATH)
    des = []
    kp = []
    matches_between_pairs = []
    point_clouds = []
    max_pix_deviation = 2 #  from rectified stereo pattern
    for i in range(2):
        logger.info("* Processing image {}".format(i))
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
        logger.info("\tFinished processing key points, descriptors and matches for image {}".format(i))

        fig = plt.figure("3D points pair {}".format(i))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='o', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    logger.info("\n")
    logger.info("* Matching key points of successive left images")
    matches_frame0_frame1 = match_descriptors_knn(des[FRAME0][LEFT], des[FRAME1][LEFT], k=2)
    good_matches_between_frames, bad_matches_between_frames = filter_matches_by_significance_test(matches_frame0_frame1, ratio_th=0.7)

    # filter keypoints that are matched between left-right pairs and between frames:
    good_matches = consistent_matches_between_successive_frames(kp, 
                                                                    good_matches_between_frames,
                                                                    matches_between_pairs)

    good_matches = np.array(good_matches)
    logger.info("\tFinished matching key points of successive left images")
    logger.info("\tWe now have matches of same point between 2 successive stereo pairs")
    logger.info("\n")

    ##############################################################################
    ####################### Displaying matches between frames ####################
    ##############################################################################
    # good mathes: [FRAME0 matches, FRAME1 matches]
    # matches on first image pair:
    fig, axes = plt.subplots(2, 1)
    
       
    img1, img2 = read_image_pair(DATA_PATH, 0)
    frame0_img = cv2.drawMatches(img1, kp[FRAME0][LEFT], img2, kp[FRAME0][RIGHT], good_matches[:5,FRAME0], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes.ravel()[0].imshow(frame0_img)
    

    img3, img4 = read_image_pair(DATA_PATH, 1)
    frame1_img = cv2.drawMatches(img3, kp[FRAME1][LEFT], img2, kp[FRAME1][RIGHT], good_matches[:5,FRAME1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    axes.ravel()[1].imshow(frame1_img)
    plt.tight_layout()


    ############################################################
    #  PNP of 4 keypoints that were matched on all four images #
    ############################################################
    logger.info("* PNP of 4 keypoints that were matched on all four images")
    point_clouds = []
    
    X = linear_least_squares_triangulation(K@P, kp[FRAME0][LEFT], K@Q, kp[FRAME0][RIGHT], good_matches[:,FRAME0]).transpose()
    point_clouds.append(X[:3, :])

    X = linear_least_squares_triangulation(K@P, kp[FRAME1][LEFT], K@Q, kp[FRAME1][RIGHT], good_matches[:,FRAME1]).transpose()
    point_clouds.append(X[:3, :])

    # randomlly select 4 keypoints from the goo7d matches
    random_indices = np.random.choice(len(good_matches), 4, replace=False)
    
    # calculate PnP:
    success, r1, t1 = cv2.solvePnP(objectPoints=np.array([point_clouds[FRAME0][:,i] for i in random_indices]),
                                   imagePoints=np.array([kp[FRAME1][LEFT][good_matches[:, FRAME1][i].queryIdx].pt for i in random_indices]),
                                   cameraMatrix=K,
                                   distCoeffs=np.zeros((4,1)),
                                   flags=cv2.SOLVEPNP_P3P)
    R, _ = cv2.Rodrigues(r1)
    trans = -R.transpose() @ t1
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

    pos1_left = pos0_left + trans.transpose()
    pos1_right = pos0_right + trans.transpose()

    pos_left = np.vstack((pos0_left, pos1_left))
    pos_right = np.vstack((pos0_right, pos1_right))

    plt.figure('relative camera positions in world coordinates')
    plt.scatter(pos_left[:, 0], pos_left[:, -1], color='r', marker='x', alpha=0.5, label='left camera')
    plt.scatter(pos_right[:, 0], pos_right[:, -1], color='b', marker='o', alpha=0.5, label='right camera')
    plt.xlabel('x[m]')
    plt.ylabel('z[m]')
    plt.legend()

    ####################################################
    # Get supporters of the transform found with PnP
    ####################################################
    logger.info("* Get supporters of the transform found with PnP")
    T = np.hstack((R, t1))
    # R = rodriguez_to_mat(r1, np.zeros_like(t1))
    # T = np.hstack((R.transpose(), -R.transpose()@t1))
    supporters_idx = get_transformation_supporters(K, P, Q, T, kp, good_matches, point_clouds[FRAME0], px_dist=2)
    logger.info("\t found {} supporters".format(len(supporters_idx)))

    #############################################################
    #### Perform full RANSAC to find the best transformation ####
    #############################################################
    logger.info("* Perform full RANSAC to find the best transformation")
    RANSAC_iterations = 100
    RANSAC_threshold = 2 # in pixels

    T, supporters_idx = Ransac(K, P, Q, kp, good_matches, point_clouds, RANSAC_iterations, RANSAC_threshold)
                    
    
    # transforming world points to point cloud in FRAME1 coordinate systems:
    X = transform_point_cloud(point_clouds[FRAME0], T)

    fig = plt.figure("Transforming frame 0 point cloud onto frame1")
    # ax = fig.add_subplot(111, projection='2d')
    # ax.scatter(point_clouds[FRAME1][0,:], point_clouds[FRAME1][1,:], point_clouds[FRAME1][2,:], c='r', marker='o', alpha=0.5)
    # ax.scatter(X[0,:], X[1,:], X[2,:], c='b', marker='^', alpha=0.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax = fig.add_subplot(111)
    ax.scatter(point_clouds[FRAME1][0,:], point_clouds[FRAME1][2,:], c='r', marker='o', alpha=0.5, label='frame1')
    ax.scatter(point_clouds[FRAME0][0,:], point_clouds[FRAME0][2,:], c='g', marker='o', alpha=0.5, label='frame0')
    ax.scatter(X[0,:], X[2,:], c='b', marker='^', alpha=0.5, label='frame0 transformed')
    
    ax.set_xlabel('X')
    ax.set_xlim([-20, 10])
    ax.set_ylabel('Z')
    ax.set_ylim([0, 30])
    ax.set_aspect('equal')
    ax.legend()

    return

if __name__ == '__main__':
    logger.info("\n")
    logger.info("*** Start ex3 ***")

    K, P, Q = read_cameras(DATA_PATH)
    ground_truth_poses = read_ground_truth_camera_pose(DATA_PATH)
    
    NUM_IMAGES = len(ground_truth_poses)
    
    # example of steps performed only on first two frames:
    first_two_frames()

    ########################################################
    ####  FINDING CAMERA POSITION OVER ENTIRE DATA SET #####
    ########################################################
    logger.info("* FINDING CAMERA POSITION OVER ENTIRE DATA SET")
    # collect keypoints, descriptors and matches for all image pairs:
    
    des = []
    kp = []
    matches_between_pairs = []
    max_pix_deviation = 2 #  from rectified stereo pattern
    logger.info("* Start collecting keypoints, descriptors and matches for all image pairs")  
    for i in range(NUM_IMAGES):
        logger.info("\t Collecting keypoints from image pair: {}".format(i))
        img1, img2 = read_image_pair(DATA_PATH, i)
        kp1, des1 = orb_detect_and_compute(img1)
        kp2, des2 = orb_detect_and_compute(img2)
        # match descriptors
        curent_pair_matches = [m[0] for m in match_descriptors_knn(des1, des2, k=2)]
        good_matches, bad_matches = filter_matches_by_rectified_stereo_pattern_constraint(kp1, kp2, curent_pair_matches, max_rect_deviation=max_pix_deviation)
        
        kp.append((kp1, kp2))
        des.append((des1, des2))
        matches_between_pairs.append((good_matches))

    logger.info("* Finished collecting image pairs")
    logger.info("* Start finding consistent matches between successive frames")
    good_matches = []
    point_clouds = []
 
    lost_frames = []

    T_all = [np.eye(4)]
    left_cam_poses = [np.eye(4)]
    RANSAC_iterations = 500
    RANSAC_threshold = 3 # in pixels

    logger.info("* Matching key points of successive left images")
    start_time = time.time()
    # updating frame 0
    left_pose = get_camera_pose(T_all[-1])
    left_cam_poses_txt = left_pose.astype(np.float16).flatten().tolist()
    with open("../outputs/ex3/poses.txt", '+a') as f:
        f.write(" ".join(map(str, left_cam_poses_txt)) + "\n")

    with open("../outputs/ex3/pnp_relative_poses.txt", "+a") as f:
                f.write(" ".join(map(str, T_all[0][:-1,:].astype(np.float16).flatten().tolist())) + "\n")

    for i in range(1, NUM_IMAGES):
        Frame0 = i-1
        Frame1 = i
        
        matches_frame0_frame1 = match_descriptors_knn(des[Frame0][LEFT], des[Frame1][LEFT], k=2)
        good_matches_between_frames, bad_matches_between_frames = filter_matches_by_significance_test(matches_frame0_frame1, ratio_th=0.8)

        # filter keypoints that are matched between left-right pairs and between frames:
        curr_good_matches = consistent_matches_between_successive_frames(kp, 
                                                                        good_matches_between_frames,
                                                                        matches_between_pairs[Frame0:Frame1+1])

        good_matches.append(np.array(curr_good_matches))

        X1 = linear_least_squares_triangulation(K@P, kp[Frame0][LEFT], K@Q, kp[Frame0][RIGHT], good_matches[-1][:,FRAME0]).transpose()
        X2 = linear_least_squares_triangulation(K@P, kp[Frame1][LEFT], K@Q, kp[Frame1][RIGHT], good_matches[-1][:,FRAME1]).transpose()
        point_clouds.append((X1[:3, :], X2[:3, :]))

        logger.info("Frames: {}-{}".format(Frame0, Frame1))
        logger.info("Finding transsformation of frame {} relative to frame 0 coordinates: Running RANSAC".format(Frame1))
        try:
            T, supporters_idx = Ransac(K, P, Q, kp[Frame0:Frame1+1], good_matches[-1], point_clouds[-1], RANSAC_iterations, RANSAC_threshold)
            logger.info("Ransac result: {}".format(T))
            T = np.vstack((T, np.array([0,0,0,1])))
            T_all.append(T_all[-1] @ T)
                        
            left_pose = get_camera_pose(T_all[-1])
            left_cam_poses_txt = left_pose.astype(np.float16).flatten().tolist()
            relative_pose = get_camera_pose(T[:-1,:]).astype(np.float16).flatten().tolist()
            with open("../outputs/ex3/poses.txt", '+a') as f:
                f.write(" ".join(map(str, left_cam_poses_txt)) + "\n")

            with open("../outputs/ex3/pnp_relative_poses.txt", "+a") as f:
                f.write(" ".join(map(str, relative_pose)) + "\n")
                
                
            left_cam_poses.append(left_pose)
            dt = np.linalg.norm(left_cam_poses[Frame1][:3, :] - left_cam_poses[Frame1-1][:3, :])
            
        except Exception as e:
            logger.info(e)
            lost_frames.append(Frame1)
        except ValueError as e:
            logger.info(e)
            lost_frames.append(Frame1)

    end_time = time.time()
    logger.info("*** COLLECTED {} CAMERA EXTRINSIC TRANSFORMS ***".format(len(T_all)))
    
    total_exec_time = end_time - start_time
    min, sec = divmod(total_exec_time, 60)
    logger.info("*** TOTAL ELAPSED TIME: %d[min] %.3f[sec]",int(min), sec)

    logger.info("*** LOST FRAMES: {}".format(lost_frames))

    xz_estimated = np.array([(pose[0,-1], pose[2,-1]) for pose in left_cam_poses[:NUM_IMAGES]])
    xz_gt = np.array([(pose[0,-1], pose[2,-1]) for pose in ground_truth_poses[:NUM_IMAGES]])

    plt.figure('left camera trajectory ')
    plt.scatter(xz_estimated[:,0], xz_estimated[:,1], color='r', marker='x', alpha=0.5, label='left camera')
    plt.scatter(xz_gt[:,0], -xz_gt[:,1], color='b', marker='o', alpha=0.5, label='ground_truth')
    plt.xlabel('x[m]')
    plt.ylabel('z[m]')
    plt.legend()
    
    os.makedirs("../outputs/ex3", exist_ok=True)
    plt.savefig('../outputs/ex3/left_cam_trajectory.png')

plt.show()


    
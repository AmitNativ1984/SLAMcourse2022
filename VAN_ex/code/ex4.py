from cProfile import label
import cv2
import numpy as np
from data_utils import *
from tracking.feature_tracking import *
from tracking.tracker import Tracker, frames_belonging_to_track, tracks_belonging_to_frame, get_features_locations
from triangulation import *
from params import *
import time
import logging
logging.basicConfig(format='%(asctime)s [%(name)s] [%(funcName)s] [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S ', level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import os
import pandas as pd

BUILT_DATABASE = False
database_file_name = 'landmarksSparse'
database_file_path = '../outputs/ex4'

def mark_feature_and_cut_patch(image, x, y, patch_size):
    """Cut a patch of size patch_size from image at position (x,y)

    Args:
        image (numpy.ndarray): image
        x (int): x coordinate of the center of the patch
        y (int): y coordinate of the center of the patch
        patch_size (int): size of the patch

    Returns:
        numpy.ndarray: patch
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 2)

    x0 = int(x - patch_size / 2)
    x1 = int(x + patch_size / 2)
    
    y0 = int(y - patch_size / 2)
    y1 = int(y + patch_size / 2)

    if x0 < 0:
        x0 = 0
        x1 = patch_size + x0
    elif x1 > image.shape[1]:
        x1 = image.shape[1]
        x0 = x1 - patch_size

    if y0 < 0:
        y0 = 0
        y1 = patch_size + y0
    elif y1 > image.shape[0]:
        y1 = image.shape[0]
        y0 = y1 - patch_size
    
    return image[y0:y1, x0:x1]


if __name__ == '__main__':
    logger.info("\n")
    logger.info("*** START EX4 ***")

    tracker = Tracker()

    if BUILT_DATABASE:
        # Load data
        ground_truth_poses = read_ground_truth_camera_pose(DATA_PATH)
        NUM_IMAGES = len(ground_truth_poses)
        logger.info("Number of images: {}".format(NUM_IMAGES))
        logger.info(" Collecting data...")
        des = []
        kp = []
        matches_between_pairs = []
        max_pix_deviation = 2 #  from rectified stereo pattern
        matches_between_frames = []
        
        for currFrame in range(NUM_IMAGES):
            logger.info("Collecting keypoints from frame {}".format(currFrame))
            img1, img2 = read_image_pair(DATA_PATH, currFrame)
            kp1, des1 = orb_detect_and_compute(img1)
            kp2, des2 = orb_detect_and_compute(img2)
            # match descriptors
            curent_pair_matches = [m[0] for m in match_descriptors_knn(des1, des2, k=2)]
            good_matches, bad_matches = filter_matches_by_rectified_stereo_pattern_constraint(kp1, kp2, curent_pair_matches, max_rect_deviation=max_pix_deviation)
            
            kp.append((kp1, kp2))
            des.append((des1, des2))
            matches_between_pairs.append((good_matches))
            
            logger.info("Found {} key points".format(len(good_matches)))

            ####################################################
            ### CONSENSUS MATCHING BETWEEN SUCCESSIVE FRAMES ###
            ####################################################
            if currFrame == 0: # need 2 frames for consensus matching
                continue

            prevFrame = currFrame - 1
            matches_prevFrame_currFrame = match_descriptors_knn(des[prevFrame][LEFT], des[currFrame][LEFT], k=2)
            matches_curr_frames, _ = filter_matches_by_significance_test(matches_prevFrame_currFrame, ratio_th=0.8)
            logger.info(f"Found {len(matches_curr_frames)} matches between frame #{prevFrame} and frame #{currFrame}")

            # filter keypoints that are matched between that are consistant between left-right pairs, and between successive frames
            matches_between_frames.append(np.array(consistent_matches_between_successive_frames(kp, 
                                                                                                matches_curr_frames,
                                                                                                matches_between_pairs[prevFrame:currFrame+1])))

        
            #                                 pair(prev_frame)      pair(curr_frame)
            #                               -------------------  --------------------
            #   matches_between_frames[i] = [(kp_left_idx, kpt_right_idx), (kp_left_idx, kpt_right_idx)]
            #                       kp[i] = [(kp_left_img, kpt_right_img)]
            tracker.add_frame_pair(kp[-2:], matches_between_frames[-1])

            # dropping first row of database which is NAN
            if currFrame == 1:
                tracker.database =tracker.database.iloc[1:,:].reset_index(drop=True)
        
        # save data
        logger.info("Saving data...")
        # serlialize database to pickle:   
        os.makedirs(database_file_path, exist_ok=True)
        tracker.database.to_pickle(os.path.join(database_file_path, database_file_name + '.pkl'))

    database = pd.read_pickle(os.path.join(database_file_path, database_file_name + '.pkl'))
   
    ############################
    ### 4.2 TRACK STATISTICS ###
    ############################

    # number of toal tracks and frames:
    num_tracks = database.shape[0]
    num_frames = int(database.shape[1] / 4)
    logging.info("Total number of tracks: {}".format(num_tracks))
    logging.info("Total number of frames: {}".format(num_frames))

    track_lengths = np.array(database.count(axis='columns'))//4
    logging.info("Average track length: {}".format(int(np.mean(track_lengths))))
    logging.info("Max track length: {}".format(np.max(track_lengths)))
    logging.info("Min track length: {}".format(np.min(track_lengths)))
    
    avg_frame_links = int(np.mean(np.array(database.count(axis='rows'))[0:-1:4]))
    logging.info("Average number of links per frame: {}".format(avg_frame_links))

    track_ids_in_frame = tracks_belonging_to_frame(database, 0)
    frames_ids_of_track = frames_belonging_to_track(database, 0)
    xlxry = get_features_locations(database, 10, 0)

    ###############################
    ### 4.3 TRACK VISUALIZATION ###
    ###############################

    # pick a random track with length > 10:
    track_id = np.random.choice(np.where(track_lengths > 10)[0])
    logging.info("Picked track id: {}".format(track_id))   
    logging.info("Track length: {}".format(track_lengths[track_id]))
    logging.info("Frames ids: {}".format(frames_belonging_to_track(database, track_id)))
    
    # cropping left and right images with patches 100x100 pixels:
    patches = []
    frames = frames_belonging_to_track(database, track_id)
    for frame_id in frames:
        img1, img2 = read_image_pair(DATA_PATH, frame_id)
        xl_xr_y = get_features_locations(database, track_id, frame_id)
        xl = xl_xr_y[0]
        xr = xl_xr_y[1]
        y = xl_xr_y[2]
        
        left_img_patch = mark_feature_and_cut_patch(img1, xl, y, patch_size=100)
        cv2.imshow('left_img_patch', left_img_patch)
        right_img_patch = mark_feature_and_cut_patch(img2, xr, y, patch_size=100)
        cv2.imshow('right_img_patch', right_img_patch)
        cv2.waitKey(1)
        left_right_patch = np.hstack((left_img_patch, right_img_patch))
        patches.append(left_right_patch)

    # plot patches:
    fig, ax = plt.subplots(len(patches), 1, figsize=(20, 20))
    for i in range(len(patches)):
        ax[i].imshow(cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
        txt = "Frame id: {}".format(frames[i])
        ax[i].text(0,0,txt)
        ax[i].axis('off')
    
    fig.set_size_inches([5, 20])
    fig.tight_layout()
    plt.savefig(os.path.join(database_file_path, 'track_patches.png'))

    ###############################
    ### 4.4 CONNECTIVITY GRAPH  ###
    ###############################
    # for every frame, present the number of frames outgoing to the next frame

    frame_links = []
    for frame_id in range(num_frames-1):
        cropped_db = database.iloc[:, 4*frame_id:4*(frame_id+2)]
        num_links = cropped_db[cropped_db.count(axis='columns')==8].shape[0]
        frame_links.append(num_links)
        logging.info("Frame id: {} Number of links: {}".format(frame_id, num_links))

    # plot frame links:
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(frame_links)
    ax.plot([0, len(frame_links)], [avg_frame_links, avg_frame_links], 'r--')
    ax.set_xlabel('[#] Frame id')
    ax.set_ylabel('[#] outgoing tracks')
    ax.set_title('Connectivity graph')
    plt.savefig(os.path.join(database_file_path, 'frame_links.png'))

    ###############################
    ### 4.6 TRACK LENGTH HISTO  ###
    ###############################
    # plot track length histogram:
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.hist(track_lengths, bins=list(range(0, np.max(track_lengths))))
    ax.set_xlabel('[#] Track length')
    ax.set_ylabel('[#] Number of tracks')
    ax.set_title('Track length histogram')
    plt.savefig(os.path.join(database_file_path, 'track_length_hist.png'))

    ###############################
    ### 4.7 REPROJECTION ERROR  ###
    ###############################
    # GT camera matrices of frames including the picked track:
    K, P, Q = read_cameras(DATA_PATH)
    ground_truth_poses = read_ground_truth_camera_pose(DATA_PATH)
    gt_poses = [ground_truth_poses[i] for i in frames]
    
    # dimentionality reduction matrix:
    A = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]])

    P_ = np.vstack((P, np.array([0,0,0,1])))
    Q_ = np.vstack((Q, np.array([0,0,0,1])))

    T = gt_poses[-1]
    T_ = np.vstack((T, np.array([0,0,0,1])))

    # traingulate track on last frame in 3D world coordinates:
    xl_xr_y = get_features_locations(database, track_id, frames[-1])    
    xl = xl_xr_y[0]
    xr = xl_xr_y[1]
    y = xl_xr_y[2]
    X_last_frame = traingulate(K@A@P_@T_, xl, y, K@A@Q_@T_, xr, y).reshape(4,1)
    logging.info("3D point of track in last frame: {}".format(X_last_frame[:3]))

    reproj_error_left_cam = []
    reproj_error_right_cam = []
    patches = []
    for i, frame_id in enumerate(frames):
        img1, img2 = read_image_pair(DATA_PATH, frame_id)
        xl_xr_y = get_features_locations(database, track_id, frame_id)
        xl = xl_xr_y[0]
        xr = xl_xr_y[1]
        y = xl_xr_y[2]
        
        T = gt_poses[i]
        T_ = np.vstack((T, np.array([0,0,0,1])))

        # reproject the 3D point on the image:
        px_l_reproj = K@A@P_@T_@X_last_frame
        xl_rep = int(px_l_reproj[0]/px_l_reproj[2])
        yl_rep = int(px_l_reproj[1]/px_l_reproj[2])
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img1 = cv2.circle(img1, (int(xl_rep), int(yl_rep)), 2, (255, 255, 0), 2)
        left_img_patch = mark_feature_and_cut_patch(img1, xl, y, patch_size=100)
        
        px_r_reproj = K@A@Q_@T_@X_last_frame
        xr_rep = int(px_r_reproj[0]/px_r_reproj[2])
        yr_rep = int(px_r_reproj[1]/px_r_reproj[2])
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2 = cv2.circle(img2, (int(xr_rep), int(yr_rep)), 2, (255, 255, 0), 2)
        right_img_patch = mark_feature_and_cut_patch(img2, xr, y, patch_size=100)

        cv2.imshow('left_img_patch', left_img_patch)
        cv2.imshow('right_img_patch', right_img_patch)
        cv2.waitKey(1)
        left_right_patch = np.hstack((left_img_patch, right_img_patch))
        patches.append(left_right_patch)
        
        # calculate reprojection error:
        reproj_error_left_cam.append(np.linalg.norm(np.array([xl, y]) - np.array([xl_rep, yl_rep])))
        reproj_error_right_cam.append(np.linalg.norm(np.array([xr, y]) - np.array([xr_rep, yr_rep])))

    # plot patches:
    fig, ax = plt.subplots(len(patches), 1, figsize=(20, 20))
    for i in range(len(patches)):
        ax[i].imshow(cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
        txt = "Frame id: {}".format(frames[i])
        ax[i].text(0,0,txt)
        ax[i].axis('off')
    
    fig.set_size_inches([5, 20])
    fig.tight_layout()
    plt.savefig(os.path.join(database_file_path, 'track_reproj_from_last_frame.png'))
    
    # plot reprojection error:
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(reproj_error_left_cam, label='left camera')
    ax.plot(reproj_error_right_cam, label='right camera')
    ax.set_xlabel('[#] Frame id')
    ax.set_ylabel('[px] Reprojection error')    
    ax.set_title('Reprojection error - Last frame to first frame')
    ax.legend()
    plt.savefig(os.path.join(database_file_path, 'track_reproj_err_from_last_frame_plot.png'))
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot reprojection error from the first frame in the track to the last:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    T = gt_poses[0]
    T_ = np.vstack((T, np.array([0,0,0,1])))

    # traingulate track on last frame in 3D world coordinates:
    xl_xr_y = get_features_locations(database, track_id, frames[0])    
    xl = xl_xr_y[0]
    xr = xl_xr_y[1]
    y = xl_xr_y[2]
    X_first_frame = traingulate(K@A@P_@T_, xl, y, K@A@Q_@T_, xr, y).reshape(4,1)
    logging.info("3D point of track in last frame: {}".format(X_first_frame[:3]))

    reproj_error_left_cam = []
    reproj_error_right_cam = []
    patches = []
    for i, frame_id in enumerate(frames):
        img1, img2 = read_image_pair(DATA_PATH, frame_id)
        xl_xr_y = get_features_locations(database, track_id, frame_id)
        xl = xl_xr_y[0]
        xr = xl_xr_y[1]
        y = xl_xr_y[2]
        
        T = gt_poses[i]
        T_ = np.vstack((T, np.array([0,0,0,1])))

        # reproject the 3D point on the image:
        px_l_reproj = K@A@P_@T_@X_first_frame
        xl_rep = int(px_l_reproj[0]/px_l_reproj[2])
        yl_rep = int(px_l_reproj[1]/px_l_reproj[2])
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img1 = cv2.circle(img1, (int(xl_rep), int(yl_rep)), 2, (255, 255, 0), 2)
        left_img_patch = mark_feature_and_cut_patch(img1, xl, y, patch_size=100)
        
        px_r_reproj = K@A@Q_@T_@X_first_frame
        xr_rep = int(px_r_reproj[0]/px_r_reproj[2])
        yr_rep = int(px_r_reproj[1]/px_r_reproj[2])
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2 = cv2.circle(img2, (int(xr_rep), int(yr_rep)), 2, (255, 255, 0), 2)
        right_img_patch = mark_feature_and_cut_patch(img2, xr, y, patch_size=100)

        cv2.imshow('left_img_patch', left_img_patch)
        cv2.imshow('right_img_patch', right_img_patch)
        cv2.waitKey(1)
        left_right_patch = np.hstack((left_img_patch, right_img_patch))
        patches.append(left_right_patch)
        
        # calculate reprojection error:
        reproj_error_left_cam.append(np.linalg.norm(np.array([xl, y]) - np.array([xl_rep, yl_rep])))
        reproj_error_right_cam.append(np.linalg.norm(np.array([xr, y]) - np.array([xr_rep, yr_rep])))

    # plot patches:
    fig, ax = plt.subplots(len(patches), 1, figsize=(20, 20))
    for i in range(len(patches)):
        ax[i].imshow(cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))
        txt = "Frame id: {}".format(frames[i])
        ax[i].text(0,0,txt)
        ax[i].axis('off')
    
    fig.set_size_inches([5, 20])
    fig.tight_layout()
    plt.savefig(os.path.join(database_file_path, 'track_reproj_from_first_frame.png'))
    
    # plot reprojection error:
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(reproj_error_left_cam, label='left camera')
    ax.plot(reproj_error_right_cam, label='right camera')
    ax.set_xlabel('[#] Frame id')
    ax.set_ylabel('[px] Reprojection error')    
    ax.set_title('Reprojection error - First frame to last frame')
    ax.legend()
    plt.savefig(os.path.join(database_file_path, 'track_reproj_err_from_first_frame_plot.png'))
        

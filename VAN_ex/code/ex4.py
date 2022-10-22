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
    
    frame_links = np.array(database.count(axis='rows'))[0:-1:4]
    logging.info("Average number of links per frame: {}".format(int(np.mean(frame_links))))

    track_ids_in_frame = tracks_belonging_to_frame(database, 0)
    frames_ids_of_track = frames_belonging_to_track(database, 0)
    xlxry = get_features_locations(database, 10, 0)

    
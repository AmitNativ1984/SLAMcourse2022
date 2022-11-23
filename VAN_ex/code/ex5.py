import cv2
import numpy as np
from data_utils import *
from tracking.feature_tracking import *
from params import *
from tracking.tracker import frames_belonging_to_track, tracks_belonging_to_frame, get_features_locations

import logging
logging.basicConfig(format='%(asctime)s [%(name)s] [%(funcName)s] [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S ', level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import os
import pandas as pd


if __name__ == '__main__':
    logging.info(" *** START EX 5 ***")
    # Load data
    logging.info("Loading data...")
    database = pd.read_pickle(LANDMARK_DATA_PATH)

    pnp_poses = read_ground_truth_camera_pose('../outputs/ex3/poses.txt', redirect_path=False)

    num_tracks = database.shape[0]
    num_frames = int(database.shape[1] / 4)
    logging.info("Total number of tracks: {}".format(num_tracks))
    logging.info("Total number of frames: {}".format(num_frames))

    track_lengths = np.array(database.count(axis='columns'))//4
    
    # Randomly select track of length >= 10:
    track_id = np.random.choice(np.where(track_lengths > 10)[0])
    logging.info("Picked track id: {}".format(track_id))   
    logging.info("Track length: {}".format(track_lengths[track_id]))
    frames_with_track = frames_belonging_to_track(database, track_id)
    logging.info("Frames ids: {}".format(frames_with_track))

    # loading camera posese from pnp:
    poses = pnp_poses[frames_with_track]


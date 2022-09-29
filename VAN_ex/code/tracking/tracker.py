import pandas as pd
import numpy as np
import os
from params import *


class Tracker():
    """Build data base that tracks with track_id over all frames

        Dataframe structure: {pandas.SparseDataFrame}
            rows: track_id
            columns: frame_id

        Dataframe[track_id][frame_id] = (kpt_left, kpt_right) --> the keypoint of #track_id in #frame_id

    """

    def __init__(self) -> None:
        self._next_track_id = 0
        
        # create sparse dataframe for all dataset - starting from two empty, sparse frames
        self.database = pd.DataFrame(pd.Series([], dtype='object'))
        self.spdtype = self.database.dtypes[0]

        self.last_frame_id = 0

    def add_frame_pair(self, kpts, matches_between_frames):
        """Update tracker with new frame

        Args:
            frame1_idx (int): first frame index
            frame2_idx (int): second frame index
            matches_successive_frames (list): list of matches between frame1 and frame2
            
            matches are indices to kpts.
            kpts are the kpts themselves per image pair (left & right images)
            
            #                                 pair(prev_frame)      pair(curr_frame)
            #                               -------------------  --------------------
            #   matches_between_frames[i] = [(kp_left_idx, kpt_right_idx), (kp_left_idx, kpt_right_idx)]
            #                       kp[i] = [(kp_left_img, kpt_right_img)]
        """

        # adding a new frame to database:
        self.database[self.last_frame_id+1] = np.nan
        
        # convert incoming frames to pandas series with each item [tack_id,frame_id] = ((u,v)_left, (u,v)_right)
        incoming_pixels_DF = self.convert_matches_to_DataFrame(kpts, matches_between_frames)
       
        # 
        unmatched_pixels_DF = self.match(incoming_pixels_DF)
        self.initiate_new_tracks(unmatched_pixels_DF)
        
        self.last_frame_id += 1
        self._next_track_id = self.database.shape[0]

    def convert_matches_to_DataFrame(self, kpts, matches_between_frames):
        # frame0_ktps_pixels = pd.Series().astype(pd.SparseDtype("float", np.nan))
        # frame1_kpts_pixels = pd.Series().astype(pd.SparseDtype("float", np.nan))
        frame0_ktps_pixels = []
        frame1_kpts_pixels = []
        
        for match in matches_between_frames:
            # frame1
            kp_left_frame0 = kpts[FRAME0][LEFT][match[FRAME0].queryIdx]
            kp_right_frame0 =  kpts[FRAME0][RIGHT][match[FRAME0].trainIdx]
            frame0_ktps_pixels.append([kp_left_frame0.pt, kp_right_frame0.pt])

            # frame2
            kp_left_frame1 = kpts[FRAME1][LEFT][match[FRAME1].queryIdx]
            kp_right_frame1 = kpts[FRAME1][RIGHT][match[FRAME1].trainIdx]
            frame1_kpts_pixels.append([kp_left_frame1.pt, kp_right_frame1.pt])

        incoming_kpt_pixels = pd.DataFrame()
        incoming_kpt_pixels[0] = pd.Series(frame0_ktps_pixels)
        incoming_kpt_pixels[1] = pd.Series(frame1_kpts_pixels)
        return incoming_kpt_pixels

    def match(self, unmatched_pixels_DF: pd.DataFrame)->pd.DataFrame:
        """Match new frame with tracks in previous recorded frame in data base.
            check kpts ids!!

        Args:
            matches (list): kpt matches in the frame to be checked.
        
        Returns:
            unmatched_pixels_DF (pandas.DataFrame): dataframe of unmatched pixels
        """

        

        # A match happens when a first frame in the new match frame pair, matches the same point in the last frame in the database
        active_tracks_ids = self.database[self.database.iloc[:, self.last_frame_id].notna()][1].index

        # Iterate over all active tracks from the last frame. 
        # Match them with the first frame in the new match frame pair (that is the same kpts that where active in prev frame pair, and also 
        # in the current frame pair)
        self.database[self.last_frame_id+1] = self.database[self.last_frame_id+1].astype('object')
        for track_id in active_tracks_ids.to_list():
            track_pixels = self.database.iloc[track_id, self.last_frame_id][LEFT]
            # find if same pixels exits in the new frame pair:
            matched_pixels_mask = unmatched_pixels_DF[FRAME0].apply(lambda x: np.all(np.array(np.array(x) == np.array(track_pixels))[LEFT])).to_list()
            
            # update track_id in next frame, with the new pixels
            matched_pixels_idx = np.where(matched_pixels_mask)
            if matched_pixels_idx[0].size == 0:
                continue
            
            matched_pixels_idx = matched_pixels_idx[0][0]
            self.database[self.last_frame_id + 1][track_id] = unmatched_pixels_DF[FRAME1][matched_pixels_idx]

            # drop the matched index from the unmatched_pixels_DF
            unmatched_pixels_DF.drop(unmatched_pixels_DF.index[matched_pixels_idx], inplace=True, axis=0)
            unmatched_pixels_DF.reset_index(drop=True, inplace=True)

        # The remaining rows in incoming_pixels_DF are unmatched tracks:
        return unmatched_pixels_DF

    def initiate_new_tracks(self, unmatched_pixels_DF):
        """Append new tracks to the database.

        Args:
            unmatched_pixels_DF (pandas.DataFrame): dataframe of unmatched pixels
        """

        to_append = pd.DataFrame(np.full([unmatched_pixels_DF.shape[0], self.database.shape[1]], np.nan))
        to_append.iloc[:,-2] = unmatched_pixels_DF[FRAME0]
        to_append.iloc[:,-1] = unmatched_pixels_DF[FRAME1]
        
        # append unmatched tracks: create new tracks in incoming frames ids:
        self.database = self.database.append(to_append, ignore_index=True)        
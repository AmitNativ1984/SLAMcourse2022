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
        self.database = self.empty_sparse_df()
        self.last_frame_id = 0

    def empty_sparse_df(self):
        nan_ = pd.Series(pd.arrays.SparseArray(np.array([np.nan])))
        sparse_db = pd.DataFrame([nan_,nan_,nan_,nan_]).astype(pd.SparseDtype("float", np.nan))
        return sparse_db.transpose()

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
        for i in range(4):
            self.database[self.database.shape[-1]] = np.nan
        self.database = self.database.astype(pd.SparseDtype("float", np.nan))
        
        # convert incoming frames to pandas series with each item [tack_id,frame_id] = ((u,v)_left, (u,v)_right)
        incoming_pixels_DF = self.convert_matches_to_DataFrame(kpts, matches_between_frames)
       
        if self.database.empty:
            unmatched_pixels_DF = incoming_pixels_DF
        else:
            unmatched_pixels_DF = self.match(incoming_pixels_DF)
        
        self.initiate_new_tracks(unmatched_pixels_DF)
        
        self.last_frame_id += 1
        self._next_track_id = self.database.shape[0]
        

    def convert_matches_to_DataFrame(self, kpts, matches_between_frames):
        # frame0_ktps_pixels = pd.Series().astype(pd.SparseDtype("float", np.nan))
        # frame1_kpts_pixels = pd.Series().astype(pd.SparseDtype("float", np.nan))
        
        for ind, match in enumerate(matches_between_frames):
            
            # frame1
            kp_left_frame0 = kpts[FRAME0][LEFT][match[FRAME0].queryIdx]
            kp_right_frame0 =  kpts[FRAME0][RIGHT][match[FRAME0].trainIdx]

            # frame2
            kp_left_frame1 = kpts[FRAME1][LEFT][match[FRAME1].queryIdx]
            kp_right_frame1 = kpts[FRAME1][RIGHT][match[FRAME1].trainIdx]
            
            if ind == 0 :
                frame0_ktps_pixels = np.array([kp_left_frame0.pt[0], kp_left_frame0.pt[1], kp_right_frame0.pt[0], kp_right_frame0.pt[1]])
                frame1_kpts_pixels = np.array([kp_left_frame1.pt[0], kp_left_frame1.pt[1], kp_right_frame1.pt[0], kp_right_frame1.pt[1]])
            else:
                frame0_ktps_pixels = np.vstack((frame0_ktps_pixels, np.array([kp_left_frame0.pt[0], kp_left_frame0.pt[1], kp_right_frame0.pt[0], kp_right_frame0.pt[1]])))
                frame1_kpts_pixels = np.vstack((frame1_kpts_pixels, np.array([kp_left_frame1.pt[0], kp_left_frame1.pt[1], kp_right_frame1.pt[0], kp_right_frame1.pt[1]])))

        incoming_kpt_pixels = pd.DataFrame(np.hstack((frame0_ktps_pixels, frame1_kpts_pixels)))
        incoming_kpt_pixels = incoming_kpt_pixels.astype(pd.SparseDtype("float", np.nan))
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
        active_tracks_ids = self.database[self.database.iloc[:, self.last_frame_id*4].notna()][1].index

        # Iterate over all active tracks from the last frame. 
        # Match them with the first frame in the new match frame pair (that is the same kpts that where active in prev frame pair, and also 
        # in the current frame pair)
        self.database.iloc[:, -4:]=self.database.iloc[:,-4:].sparse.to_dense()
        for track_id in active_tracks_ids.to_list():
            track_pixels = np.array(self.database.iloc[track_id][self.last_frame_id*4:self.last_frame_id*4+4]).reshape(2,2)[LEFT]
            
            # find the index of matching track:
            match = unmatched_pixels_DF.iloc[:,:4].loc[(unmatched_pixels_DF[0]==track_pixels[0]) & (unmatched_pixels_DF[1]==track_pixels[1])].index

            if match.empty:
                continue
           
            # add data of new match to database (only last frame is important)
            matched_pixels_idx = match[0]
            self.database.iloc[track_id, -4] = unmatched_pixels_DF.iloc[matched_pixels_idx,-4]
            self.database.iloc[track_id, -3] = unmatched_pixels_DF.iloc[matched_pixels_idx,-3]
            self.database.iloc[track_id, -2] = unmatched_pixels_DF.iloc[matched_pixels_idx,-2]
            self.database.iloc[track_id, -1] = unmatched_pixels_DF.iloc[matched_pixels_idx,-1]
            # drop the matched index from the unmatched_pixels_DF
            unmatched_pixels_DF.drop(unmatched_pixels_DF.index[matched_pixels_idx], inplace=True, axis=0)
            unmatched_pixels_DF.reset_index(drop=True, inplace=True)

        self.database.iloc[:,-4:] = self.database.iloc[:, -4:].astype(pd.SparseDtype("float", np.nan))
        # The remaining rows in incoming_pixels_DF are unmatched tracks:
        return unmatched_pixels_DF

    def initiate_new_tracks(self, unmatched_pixels_DF):
        """Append new tracks to the database.

        Args:
            unmatched_pixels_DF (pandas.DataFrame): dataframe of unmatched pixels
        """

        to_append = pd.DataFrame(np.full((unmatched_pixels_DF.shape[0], self.database.shape[1]), np.nan))
        for col in range(-8, 0):
            to_append[to_append.shape[1]+col] = unmatched_pixels_DF.iloc[:, col]
        
        
        # append unmatched tracks: create new tracks in incoming frames ids:
        self.database = self.database.append(to_append, ignore_index=True)        


def tracks_belonging_to_frame(database, frame_id):
    """return all track ids of a given frame:

    Args:
        database (pandas): database containing all frames and tracks
        frame_id (int): frame_id 
    """

    track_ids = pd.arrays.SparseArray(database[frame_id]).sp_index.indices
    return track_ids

def frames_belonging_to_track(database, track_id):
    """return all frames of a given track

    Args:
        database (pandas): database containing all frames and tracks
        track_id (int): track id
    """
    frame_ids = pd.arrays.SparseArray(database.iloc[track_id, 0:-1:4]).sp_index.indices
    return frame_ids

def get_features_locations(database, track_id, frame_id):
    """return feature locations of track TrackId on both left and right images as a triplet (xl, xr, y)
    (xl , y) the feature location on the left image
    (xr , y) the feature location on the right image
    Note that the 𝑦 index is shared on both images.

    Args:
        database (pandas): database containing all frames and tracks
        track_id (int): track id
        frame_id (int): frame id
    """
    xl_yl_xr_yr = database.iloc[track_id, frame_id*4:frame_id*4+4] 
    return (xl_yl_xr_yr[0], xl_yl_xr_yr[2], xl_yl_xr_yr[1])
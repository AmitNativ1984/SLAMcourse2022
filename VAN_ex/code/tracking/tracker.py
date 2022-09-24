import pandas as pd
import numpy as np
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
        self._next_frame_id = 0

        # create sparse dataframe for all dataset
        self.database = pd.DataFrame(pd.Series().astype(pd.SparseDtype("float", np.nan)))

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

        # convert incoming frames to pandas series with each item [tack_id,frame_id] = ((u,v)_left, (u,v)_right)
        incoming_pixels_DF = self.convert_matches_to_DataFrame(kpts, matches_between_frames)
        
        
        #TODO: DONT FORGET TO HANDLE FRAMES CORRECTLY!!!!!!

        # TODO: UPDATE FRAME0 & FRAME1
        
        matched_tracks_idx, unmatched_tracks_idx = self.match(incoming_pixels_DF)

        # create new tracks for unmatched tracks
        # self.update_database_with_unmatched_tracks(matches_between_frames, kpts, unmatched_tracks_idx)


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
        incoming_kpt_pixels[0] = pd.Series(pd.arrays.SparseArray(frame0_ktps_pixels))
        incoming_kpt_pixels[1] = pd.Series(pd.arrays.SparseArray(frame1_kpts_pixels))
        return incoming_kpt_pixels
    
    def update_databease_with_matched_tracks(self, matches_between_frames):
        pass

    def update_database_with_unmatched_tracks(self, matches_between_frames, kpts, unmatched_tracks_idx):
        # update unmatched kpts as new tracks:
        new_tracks_frame0 = []
        new_tracks_frame1 = []
        for match in matches_between_frames[unmatched_tracks_idx]:
            # new tracks on FRAME0
            kp_left_frame0 = kpts[FRAME0][LEFT][match[FRAME0].queryIdx]
            kp_right_frame0 =  kpts[FRAME0][RIGHT][match[FRAME0].trainIdx]
            new_tracks_frame0.append((kp_left_frame0, kp_right_frame0))

            # new track on FRAME1
            kp_left_frame1 = kpts[FRAME1][LEFT][match[FRAME0].queryIdx]
            kp_right_frame1 = kpts[FRAME0][RIGHT][match[FRAME0].trainIdx]
            new_tracks_frame1.append((kp_left_frame1, kp_right_frame1))

        self.database[self._next_frame_id] = pd.Series(pd.arrays.SparseArray(new_tracks_frame0))
        self.database[self._next_frame_id+1] = pd.Series(pd.arrays.SparseArray(new_tracks_frame1))
        
        self._next_frame_id += 2
        self._next_track_id += len(unmatched_tracks_idx)

    def match(self, incoming_pixels_DF):
        """Match new frame with tracks in previous recorded frame in data base.
            check kpts ids!!

        Args:
            matches (list): kpt matches in the frame to be checked.
        
        Returns:
            matched_idx: list of matches index in the input "matches" list
            unmatched_idx: list of unmatched kpts in the input "matches" list
        """

        

        # A match happens when a first frame in the new match frame pair, matches the same point in the last frame in the database
        active_tracks = self.database.iloc[:, -1].notna()

        # Iterate over all active tracks from the last frame. 
        # Match them with the first frame in the new match frame pair (that is the same kpts that where active in prev frame pair, and also 
        # in the current frame pair)
        for track_id in active_tracks.index.to_list():
            track_pixels = self.database.iloc[track_id, -1][LEFT]
            # find if same pixels exits in the new frame pair:
            matched_idx = incoming_pixels_DF[0].apply(lambda x: x[LEFT] == track_pixels).to_list()
            self.database.iloc[self._next_frame_id + 1, track_id] = incoming_pixels_DF[FRAME1][matched_idx]

            # drop the matched index from the incoming_pixels_DF
            incoming_pixels_DF.drop(incoming_pixels_DF.index[matched_idx], inplace=True, axis=0)

        # TODO: update with correct flow

        unmatched_idx = list(range(len(incoming_pixels_DF)))

        return [], unmatched_idx

        
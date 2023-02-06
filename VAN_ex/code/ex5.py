from triangulation import invert_exterinsic_matrix
import cv2
import numpy as np
from data_utils import *
from tracking.feature_tracking import *
from params import *
from tracking.tracker import frames_belonging_to_track, tracks_belonging_to_frame, get_features_locations, features_of_all_tracks_in_frame

import logging
logging.basicConfig(format='%(asctime)s [%(name)s] [%(funcName)s] [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S ', level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import os
import pandas as pd

from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

import pickle
import time

from ex4 import  mark_feature_and_cut_patch

def gtsam_stereo_calib_model(data_path):
    # creating GTSAM stereo camera frames:
    K, P, Q = read_cameras(data_path)
    fx = K[0,0]
    fy = K[1,1]
    skew = K[0,1]
    cx = K[0,2]
    cy = K[1,2]
    baseline = Q[0,3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)

def plot_trajectory_view_from_above(values, title, fig_id = 0, marker_color='b', label=''):
    # Then 3D poses, if any
    poses_values = gtsam.utilities.allPose3s(values)
    poses = []
    for key in poses_values.keys():
        pose = poses_values.atPose3(key)
        poses.append(pose.translation())
        
    poses = np.array(poses)

    fig = plt.figure(fig_id)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel('x[m]')
    ax.set_ylabel('z[m]')
    ax.axis('equal')
    ax.set_title(title)
    ax.plot(poses[:, 0], poses[:, 2], marker_color, label=label)

def plot_points_view_from_above(values, title, fig_id=0, marker_scale=1, marker_color='b', marker_style='o', label=''):
    keys = values.keys()
    # Plot points and covariance matrices
    points = []
    for key in keys:
        try:
            point = values.atPoint3(key)
            points.append(point)
        except RuntimeError:
            continue
            # I guess it's not a Point3

    points = np.array(points)
    fig = plt.figure(fig_id)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel('x[m]')
    ax.set_ylabel('z[m]')
    ax.axis('equal')
    ax.set_title(title)    
    ax.scatter(points[:, 0], points[:, 2], marker=marker_style, s=marker_scale, color=marker_color, label=label)
    ax.legend()

def optimize_bundle_window(graph, initialEstimate, params):
    # optimize the graph:
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate, params)
    result = optimizer.optimize()
    return result

def build_bundle_window_graph(key_frames, bundle_idx, pnp_poses, initial_pose, K, measurement_noise, pose_uncertainty, database):
    # create empty graph and values:
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()

    # key frames idx:
    for frame_id in list(range(key_frames[bundle_idx], key_frames[bundle_idx+1] + 1)):
        # if first frame of bundle, relative pose id identity:
        if frame_id == key_frames[bundle_idx]:
            # insert initial pose of the camera:
            cam_pose = initial_pose

        else:
            T = pnp_poses[frame_id]
            T = np.vstack((T, np.array([0, 0, 0, 1])))
            cam_pose = np.vstack((cam_pose, np.array([0, 0, 0, 1]))) @ T
            cam_pose = cam_pose[:3, :]
        
        initialEstimate.insert(symbol('c', frame_id), gtsam.Pose3(cam_pose))           
        logging.info('Adding new camera pose to graph: c{} at world coordinates: {}'.format(frame_id, cam_pose))
        # current stereo frame:
        stereo_camera = gtsam.StereoCamera(gtsam.Pose3(cam_pose), K)
        # add initial estimate of new landmarks:
        logging.info("Collecting all tracks and features in frame: {}".format(frame_id))
        tracks = tracks_belonging_to_frame(database, frame_id)
        xl_xr_y = features_of_all_tracks_in_frame(database, frame_id, tracks)
        R = stereo_camera.pose().rotation()
        t = stereo_camera.pose().translation()
        M = np.hstack((R.transpose(), -R.transpose()@t.reshape(-1,1)))
        for idx, track_id in enumerate(tracks):
            xl, xr, y = xl_xr_y[idx]                                    
            landmark_world_point3 = stereo_camera.backproject(gtsam.StereoPoint2(xl, xr, y))

            landmark_rel_point3 = M @ np.hstack((landmark_world_point3, 1)).reshape((-1,1))
            
            if landmark_rel_point3[-1] < 0 or landmark_rel_point3[-1] >100 or \
               np.isnan(landmark_world_point3).any() or \
               np.isinf(landmark_world_point3).any():
               continue
               


            if not initialEstimate.exists(symbol('l', track_id)):
                # project the pixels in the left and right stereo frames to 3d world coordinates:
                # by using the stereo_camera model defined above:
                # For stability, reject landmarks that are two far away or behind the camera
                initialEstimate.insert(symbol('l', track_id), landmark_world_point3)
                logging.info('Adding new landmark to graph: l{} at world coordinates: {}'.format(track_id, landmark_world_point3))

            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(xl, xr, y),
                                                    measurement_noise, 
                                                    symbol('c', frame_id),
                                                    symbol('l', track_id), 
                                                    K)
            )
            logging.info("Added factor between camera pose: c{} and landmark: l{}".format(frame_id, track_id))
                
    # adding constraint to first frame:
    # graph.add(gtsam.PriorFactorPose3(symbol('c', key_frames[bundle_idx]), initialEstimate.atPose3(symbol('c', key_frames[bundle_idx])), pose_uncertainty))
    graph.add(gtsam.NonlinearEqualityPose3(symbol('c', key_frames[bundle_idx]), initialEstimate.atPose3(symbol('c', key_frames[bundle_idx]))))
    return graph, initialEstimate


if __name__ == '__main__':
    database_file_path = '../outputs/ex5'
    os.makedirs(database_file_path, exist_ok=True)
    
    logging.info(" *** START EX 5 ***")
    # Load data
    logging.info("Loading data...")
    database = pd.read_pickle(LANDMARK_DATA_PATH)

    pnp_poses = read_ground_truth_camera_pose('../outputs/ex3/poses.txt', redirect_path=False)
    # pnp_poses = read_ground_truth_camera_pose(DATA_PATH)
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
    left_cam_poses = pnp_poses[frames_with_track[0]:frames_with_track[-1]+1]
    left_cam_poses_world = [pose for pose in left_cam_poses]
    
    # creating GTSAM stereo camera frames:
    K, P, Q = read_cameras(DATA_PATH)
    fx = K[0,0]
    fy = K[1,1]
    skew = K[0,1]
    cx = K[0,2]
    cy = K[1,2]
    baseline = Q[0,3]
    K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)

    stereo_frames = []
    # creating GTSAM stereo camera frames:
    for left_cam_pose in left_cam_poses_world:
        stereo_frames.append(gtsam.StereoCamera(gtsam.Pose3(left_cam_pose), K))
    
    # traingulate the track point in the last frame:
    xl_xr_y = get_features_locations(database, track_id, frames_with_track[-1])
    logging.info("Track point in the last frame: {}".format(xl_xr_y))
    stereoPoint = gtsam.StereoPoint2(xl_xr_y[0], xl_xr_y[1], xl_xr_y[2])
    point3 = stereo_frames[-1].backproject(stereoPoint)
    logging.info("Traingulating the track point in the last frame: {} [m]".format(point3))

    # triangulating track points from the last frame to the first one:
    left_reproj_err = []
    right_reproj_err = []
    for i, frame_id in enumerate(frames_with_track):
        # projecting the point from the last frame:
        try:
            stereoPoint2 = stereo_frames[i].project(point3)        
        except Exception as e:
            logging.info("Failed to project point to frame: {}".format(frame_id))
            logging.error(e)
            continue
        xl_xr_y = get_features_locations(database, track_id, frame_id)
                
        org_pix = np.array([xl_xr_y[0], xl_xr_y[1], xl_xr_y[2]])
        left_reproj_err.append(np.linalg.norm(np.array([stereoPoint2.uL(), stereoPoint2.v()] - np.array([org_pix[0], org_pix[2]]))))
        right_reproj_err.append(np.linalg.norm(np.array([stereoPoint2.uR(), stereoPoint2.v()] - np.array([org_pix[1], org_pix[2]]))))
        
        logging.info("Frame id: {}, left reprojection error: {} [px], right reprojection error: {} [px]".format(frame_id, left_reproj_err[-1], right_reproj_err[-1]))

    # plotting left and right reprojection error:
    plt.figure()
    plt.plot(frames_with_track, left_reproj_err, label='left')
    plt.plot(frames_with_track, right_reproj_err, label='right')
    plt.legend()
    plt.title('Reprojection error, Track id: {}'.format(track_id))
    plt.savefig(os.path.join(database_file_path, 'reproj_err.png'))
        

    #####################################################
    ##        CREATING FACTOR GRAPH FOR TRACK          ##
    #####################################################
    
    # creating a factor graph container and add factors to it:
    graph = gtsam.NonlinearFactorGraph()
    
    # adding a contraint to the last pose:
    first_pose = gtsam.Pose3()
    graph.add(gtsam.NonlinearEqualityPose3(symbol('c', frames_with_track[-1]), first_pose))

    # create factor noise model with 3 sigmas, to represent real measurements:
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))

    # create and stereo factors to the graph (a single landmark, and all camera poses)
    track_landmark = symbol('l', 0)
    camera_poses = [symbol('c', i) for i in frames_with_track]

    # creating GTSAM stereo camera frames:
    stereo_frames = []
    for left_cam_pose in left_cam_poses_world:
        stereo_frames.append(gtsam.StereoCamera(gtsam.Pose3(left_cam_pose), K))
    
    
    # traingulate the track point in the last frame:
    xl_xr_y = get_features_locations(database, track_id, frames_with_track[-1])
    stereoPoint = gtsam.StereoPoint2(xl_xr_y[0], xl_xr_y[1], xl_xr_y[2])
    landmark_point3 = stereo_frames[-1].backproject(stereoPoint)        # world coordinates of track landmark
    
    # create and add stereo factors between the landmark and all camera poses:
    factors = []
    for idx, frame_id in enumerate(frames_with_track):
        xl, xr, y = get_features_locations(database, track_id, frame_id)
        factors.append(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(xl, xr, y), stereo_model, camera_poses[idx], track_landmark, K))
        graph.add(factors[-1])
        

    # create and add initial estimates:
    initalEstimate = gtsam.Values()
    initalEstimate.insert(camera_poses[-1], first_pose)
    for idx, pose in enumerate(camera_poses[:-1]):
        initalEstimate.insert(pose, gtsam.Pose3(left_cam_poses_world[idx+1]))
    
    initalEstimate.insert(track_landmark, landmark_point3)

    # factor error over the track's frames (before optimization):
    factor_error = []
    for factor in factors[:-1]:
        factor_error.append(factor.error(initalEstimate))

    # plotting factor error:
    plt.figure()
    plt.plot(frames_with_track[:-1], factor_error)
    plt.title('Factor error, Track id: {}'.format(track_id))
    plt.xlabel('Frame id')
    plt.ylabel('Factor error')
    plt.savefig(os.path.join(database_file_path, 'factor_error_before_optimization.png'))

    # plotting the factor error as function of reprojection error:
    plt.figure()
    plt.scatter(left_reproj_err[:-1], factor_error, label='left')
    plt.xlabel('Reprojection error')
    plt.ylabel('Factor error')
    plt.title('Factor error as function of reprojection error, Track id: {}'.format(track_id))
    plt.savefig(os.path.join(database_file_path, 'factor_error_vs_reproj_err_before_optimization.png'))
    
    # calculate the relative translation of pnp frames:
    relative_translations = []
    prev_pose = np.array([0, 0, 0])
    for pose in pnp_poses:
        t = pose[:, -1]
        relative_translations.append(np.linalg.norm(prev_pose - t))
        prev_pose = t

    relative_translations = np.array(relative_translations)
    # plot the relative translation of pnp frames:
    plt.figure()
    plt.plot(relative_translations)
    plt.title('Key frames')
    plt.xlabel('Frame id')
    plt.ylabel('Relative translation')

    # choosing key frames:
    key_frames, _ = find_peaks(relative_translations, distance=10)

    if key_frames[0] != 0:
        key_frames = np.insert(key_frames, 0, 0)

    if key_frames[-1] != len(relative_translations) - 1:
        key_frames = np.insert(key_frames, len(key_frames), len(relative_translations)-1)

    plt.plot(key_frames, relative_translations[key_frames], "x")
    plt.legend(["All frames", "Key frames"])
    plt.savefig(os.path.join(database_file_path, 'key_frames.png'))
    
    ###################################################
    ## Creating a factor graph for the first window  ##
    ###################################################

    # Read camera poses in world coordinates, calculate by PnP in ex3:
    # relative camera poses (M[i] = M[i-1]*T)
    pnp_poses = read_ground_truth_camera_pose('../outputs/ex3/pnp_relative_poses.txt', redirect_path=False)
    
    #Loading camera calibration parameters:
    K = gtsam_stereo_calib_model(DATA_PATH)

    # create noise models:
    projection_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))
    pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5]))
    
    # adding robust noise model with Cauchy kernel to account for outliers:
    measurement_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy(1.0), projection_uncertainty)
    
    # params for Levenberg-Marquardt optimizer:
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    params.setVerbosityLM("SUMMARY")
    
    # create empty graph and values:
    graph = gtsam.NonlinearFactorGraph()
    initalEstimate = gtsam.Values()
    initial_pose = np.eye(4)[:3, :]
    
    # # build the graph for the first window:
    # bundle_idx = 0
    # graph, initialEstimate = build_bundle_window_graph(key_frames=key_frames, 
    #                                                     bundle_idx=bundle_idx, 
    #                                                     pnp_poses=pnp_poses, 
    #                                                     initial_pose=initial_pose,
    #                                                     K=K,
    #                                                     measurement_noise=measurement_noise,
    #                                                     pose_uncertainty=pose_uncertainty)
   
    # result = optimize_bundle_window(graph, initialEstimate, params)
    
    # # print graph error prior to optimization:
    # print('Initial Error: {}'.format(graph.error(initialEstimate)))  
    # print('Final Error: {}'.format(graph.error(result)))

    # plot_trajectory_view_from_above(initialEstimate, "trajectory", fig_id=5, marker_color='green', label='initial estimate')
    # plot_trajectory_view_from_above(result, "trajectory", fig_id=5, marker_color='red', label='result')
    
    # plot_points_view_from_above(result, "trajectory", fig_id=5, marker_scale=1, marker_color='blue', marker_style='o', label='result')
    # plt.savefig(os.path.join(database_file_path, 'result_trajectory_and_points_first_bundle.png'))

    # gtsam_plot.plot_trajectory(1, initialEstimate, title="initial camera poses")
    # plt.savefig(os.path.join(database_file_path, 'initial_camera_poses_first_bundle.png'))
        
    # gtsam_plot.plot_trajectory(2, result, title="optimized camera poses")
    # plt.savefig(os.path.join(database_file_path, 'result_camera_poses_first_bundle.png'))


    ######################################
    ##  Building the entire trajectory  ##
    ######################################

    # Read camera poses in world coordinates, calculate by PnP in ex3:
    # relative camera poses (M[i] = M[i-1]*T)
    pnp_poses = read_ground_truth_camera_pose('../outputs/ex3/pnp_relative_poses.txt', redirect_path=False)
    
    #Loading camera calibration parameters:
    K = gtsam_stereo_calib_model(DATA_PATH)

    # create noise models:
    projection_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))
    pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5]))
    
    # adding robust noise model with Cauchy kernel to account for outliers:
    measurement_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy(1.0), projection_uncertainty)
    
    # params for Levenberg-Marquardt optimizer:
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    params.setVerbosityLM("SUMMARY")
    
    # create empty graph and values:
    graph = gtsam.NonlinearFactorGraph()
    initalEstimate = gtsam.Values()
    initial_pose = np.eye(4)[:3, :]
    
    initial_bundles = []
    optimized_bundles = []    

    start_time = time.time()
    for bundle_idx in range(0, len(key_frames)-1):    
        logging.info("Running bundle window: {}".format(bundle_idx))
        graph, initialEstimate = build_bundle_window_graph(key_frames=key_frames, 
                                                            bundle_idx=bundle_idx, 
                                                            pnp_poses=pnp_poses, 
                                                            initial_pose=initial_pose,
                                                            K=K,
                                                            measurement_noise=measurement_noise,
                                                            pose_uncertainty=pose_uncertainty,
                                                            database=database)
    
        initial_bundles.append(initialEstimate)

        result = optimize_bundle_window(graph, initialEstimate, params)
        optimized_bundles.append(result)
        
        # updating intial pose for next window, as the last pose from current window:
        poses_values = gtsam.utilities.allPose3s(result)
        last_keyframe_pose = poses_values.atPose3(poses_values.keys()[-1])
        initial_pose = last_keyframe_pose.matrix()[:3, :]
        
        initial_pose_label = gtsam.LabeledSymbol(poses_values.keys()[-1])
        logging.info("Initial pose for next bundle window is set to last pose of current window: {}{}".format(chr(initial_pose_label.chr()), initial_pose_label.index()))

        # saving as pickle file:
        logging.info("Saving initial and optimized bundle windows to pickle file")
        with open(os.path.join(database_file_path, 'initial_bundles.pickle'), 'wb') as handle:
            pickle.dump(initial_bundles, handle)

        with open(os.path.join(database_file_path, 'optimized_bundles.pickle'), 'wb') as handle:
            pickle.dump(optimized_bundles, handle)
        
        logging.info("Saving complete")
        
    logging.info("Calculating all bundle windows took: {} seconds".format(time.time() - start_time))
    logging.info("DONE")

    cam_pose = np.eye(4)[:3, :]
    initial_trajectory = []
    for bundle_idx in range(0, len(key_frames)-1):    
        for frame_id in list(range(key_frames[bundle_idx], key_frames[bundle_idx+1] + 1)):
            T = pnp_poses[frame_id]
            T = np.vstack((T, np.array([0, 0, 0, 1])))
            cam_pose = np.vstack((cam_pose, np.array([0, 0, 0, 1]))) @ T
            cam_pose = cam_pose[:3, :]
            initial_trajectory.append(cam_pose)
    
    # plotting optimized vs initial trajectory from above:
    for initial_bundle, optimized_bundle in zip(initial_bundles, optimized_bundles):
        # plot_trajectory_view_from_above(initial_bundle, "trajectory", fig_id='BEV trajectories', marker_color='red', label='initial estimate')
        plot_trajectory_view_from_above(optimized_bundle, "trajectory", fig_id='BEV trajectories', marker_color='blue', label='result')

    plt.legend()

    plt.show()
    
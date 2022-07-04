import numpy as np
import cv2

def linear_least_squares_triangulation(P, kp1, Q, kp2, matches):
    # P: calibration camera matrix of camera 1
    # kp1: keypoints of camera 1
    # Q: calibration camera matrix of camera 2
    # kp2: keypoints of camera 2
    # matches: matches between camera 1 kpts and camera 2 kpts (kp1, kp2)
    # return: triangulated points

    X = np.zeros((len(matches), 4))
    for idx, m in enumerate(matches):
        # Get the keypoints from the good matches
        kp1_pt = kp1[m.queryIdx].pt
        px, py = kp1_pt[0], kp1_pt[1]
        
        kp2_pt = kp2[m.trainIdx].pt
        qx, qy = kp2_pt[0], kp2_pt[1]
        
        A = np.array([P[2,:]*px - P[0,:],
                      P[2,:]*py - P[1,:],
                      Q[2,:]*qx - Q[0,:],
                      Q[2,:]*qy - Q[1,:]])

        U, S, V = np.linalg.svd(A)

        X[idx,:] = V[-1,:]/V[-1,-1]

    return X

def rodriguez_to_mat(rvec, tvec): 
  rot, _ = cv2.Rodrigues(rvec) 
  return np.hstack((rot, tvec))
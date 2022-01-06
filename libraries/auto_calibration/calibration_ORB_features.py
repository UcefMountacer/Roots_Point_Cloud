import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    
    # init ORB detector
    orb = cv2.ORB_create()

    # get features using detector
    kp, des = orb.detectAndCompute(image,None)
    
    return kp, des


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """

    display = cv2.drawKeypoints(image, kp, None)
    cv2.imwrite('features.png',display)


def match_features(des1, des2,lowe):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2) 
    
    # filter matches
    good_matches = []
    for m,n in matches:
        if m.distance < lowe * n.distance:
            good_matches.append(m)

    return matches , good_matches


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    
    image_matches = cv2.drawMatchesKnn(image1,kp1,image2,kp2,match,None,flags=2)
    cv2.imwrite('match_plus.png',image_matches)


def matches_to_calib_pts(matches, pts1, pts2):

    '''
    convert matches and keypoints into calibration points
    '''
    
    assert (len(matches) > 0)

    # print(matches)
    # print(len(matches))
    
    # print(len(pts1))
    # print(len(pts2))

    obj_pts = np.array([pts1[x.queryIdx].pt for x in matches])
    img_pts = np.array([pts2[x.trainIdx].pt for x in matches])

    # add a zero z-coordinate
    N = len(matches)
    obj_pts = np.hstack([obj_pts, np.zeros((N, 1))])

    obj_pts = obj_pts.astype(np.float32)
    img_pts = img_pts.astype(np.float32)

    return (obj_pts, img_pts)


def to_calibration_data(match_sets, obj_keys, img_keysets, min_matches):

    '''
    convert to calibration data
    '''

    match_keys = zip(match_sets, img_keysets)
    match_keys = [x for x in match_keys if len(x[0]) >= min_matches]

    calib_pts = [matches_to_calib_pts(m, obj_keys, k) \
                 for (m, k) in match_keys]

    return zip(*calib_pts)


def calibrate_intrinsic(obj_pts, img_pts, img_size):

    '''
    perform calibration using calibration data
    '''

    # [err, K, distortion, rvecs, tvecs] = cv2.calibrateCamera(obj_pts, img_pts, img_size,K,None)
    err, K, _,_,_,_,_,_,_,_ = cv2.calibrateCameraROExtended(obj_pts, img_pts, img_size, 1,None, None)

    return K, err


def mean_k(k_list, good_calib_indexes):

    '''
    to use if want to compute the mean of low error Ks instead of the minimal error K
    '''

    K_mean = np.zeros((3,3))

    K_mean[0][0] = np.mean([k_list[i][0][0] for i in good_calib_indexes])
    K_mean[0][2] = np.mean([k_list[i][0][2] for i in good_calib_indexes])
    K_mean[1][1] = np.mean([k_list[i][1][1] for i in good_calib_indexes])
    K_mean[1][2] = np.mean([k_list[i][1][2] for i in good_calib_indexes])
    K_mean[2][2] = 1

    return K_mean




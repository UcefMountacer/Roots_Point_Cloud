import numpy as np
import cv2




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

    # [err, K, distortion, rvecs, tvecs] = cv2.calibrateCamera(obj_pts, img_pts, img_size,None, None)
    err, K, D, R,t,_,_,_,_,_ = cv2.calibrateCameraROExtended(obj_pts, img_pts, img_size, 1,None, None)

    return K, err, D, R, t


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


def save_K(K,D,R,t, P1, P2, Q, file_path):

    '''
    Save the camera matrix and the distortion coefficients to given path/file.
    '''

    cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", K)
    cv_file.write("D", D)
    cv_file.write("R", R[0])
    cv_file.write("t", t[0])
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()








''' run and test '''
# import os

# def read_video(video_file_path):

#     ''' 
#     read video and return frames
#     '''

#     list_of_frames = []
#     vidcap = cv2.VideoCapture(video_file_path)

#     success,image = vidcap.read()

#     while success:
#         list_of_frames.append(image)
#         success,image = vidcap.read()

#     return list_of_frames


# video1 = 'data/MVI_0252.MOV'
# video2 = 'data/MVI_0590.MOV'

# # lowe distance between matches threshold
# filtration_threshold = 0.9

# # rms error of calibration threshold
# rms_threshold = 10.0

# # input frames from videos
# list1 = read_video(video1)
# list2 = read_video(video2)

# print('running auto-calibration using',len(list1), 'images')

# # init an empty K
# K = np.zeros((3,3))

# # define a list to store Ks with low RMS error
# k_list = []
# err_list = []
# r_list = []
# t_list = []
# d_list = []

# image_size = list1[0].shape[0] , list1[0].shape[1]

# for i, (im1,im2) in enumerate(zip(list1,list2)):

#     print('step :', i)

#     # get ORB features and their descriptors
#     kp1, des1 = extract_features(im1)
#     kp2, des2 = extract_features(im2)

#     # get matches between a pair of frames
#     _ , matches = match_features(des1, des2, filtration_threshold)


#     # calibration data operation : needed for opencv camera calibration process in this way

#     pattern_keys = list(kp1)
#     frames_keys=[list(kp1),list(kp2)]
#     frames_matches = [matches]
#     img_size = (im1.shape[1], im1.shape[0])
#     [obj_pts, image_pts] = to_calibration_data(frames_matches, pattern_keys, frames_keys, 2)


#     # get K matrix and error
#     K, err, D, R, t = calibrate_intrinsic(obj_pts, image_pts, img_size)

#     print('pair number :', i, 'with error :',err)

#     # append to list of Ks and list or respective errors
#     err_list.append(err)
#     k_list.append(K)
#     d_list.append(D)
#     r_list.append(R)
#     t_list.append(t)

# # define a list of good indexes (with good rms error)

# good_calib_indexes = []
# for i,err in enumerate(err_list):

#     if err < rms_threshold:
#         good_calib_indexes.append(i)


# # get mean of all good Ks
# # K1 = mean_k(k_list, good_calib_indexes)

# # get K with minimal error
# K = k_list[np.argmin(err_list)]
# D = d_list[np.argmin(err_list)]
# R = r_list[np.argmin(err_list)]
# t = t_list[np.argmin(err_list)]

# _, _, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K, D, K, D, image_size, R[0], t[0], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

# # print('K obtained with calculating mean of K with low RMS error: ',K1)

# print('K with the lowest RMS error :',K)

    
# save_path = os.path.join('outputs' , 'calibration_ORB_full.yml')

# save_K(K,D,R,t,P1, P2, Q, save_path)
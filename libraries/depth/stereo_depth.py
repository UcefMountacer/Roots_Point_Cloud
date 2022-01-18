import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

###############
# stereo depth
###############


'''
    IN : stereo frame
    OUT : disparity map
'''



def depth_map(left, right):
    '''
    Depth map calculation
    '''
    
    # kernel_size = 7*3
    # left = cv2.GaussianBlur(left, (kernel_size,kernel_size), sigmaX=5, sigmaY=5)
    # right = cv2.GaussianBlur(right, (kernel_size, kernel_size), sigmaX=2, sigmaY=2)
    

    stereo = cv2.StereoSGBM_create(
        numDisparities=16*50,  
        blockSize=2,
        minDisparity=0)
        
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
    # right_matcher = cv2.ximgproc.createRightMatcher(stereo)


    # FILTER Parameters
    lmbda = 8000
    sigma = 1.3

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = stereo.compute(left, right)  
    dispr = right_matcher.compute(right, left)  

    filteredImg = wls_filter.filter(displ, left, disparity_map_right=dispr)

    return filteredImg


def run_on_stereo(left , right, rectify=0, K=None):

    '''
    run code
    '''

    if rectify:

        # not working for now, may ba useful


        height, width, channel = left.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K, None, None, None, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K, None, None, None, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disp = depth_map(gray_left, gray_right) 

    if not rectify:

        # what we are using now
            
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disp = depth_map(gray_left, gray_right) 

    return disp


''' test '''

'''
def read_video(video_file_path):

    list_of_frames = []
    vidcap = cv2.VideoCapture(video_file_path)

    success,image = vidcap.read()

    while success:
        list_of_frames.append(image)
        success,image = vidcap.read()

    return list_of_frames



if __name__ == '__main__':


    # directories
    v1 = 'data/MVI_0252.MOV'
    v2 = 'data/MVI_0590.MOV'
    rectify = 0
    
    output_dir = 'depth/output'
    # input data
    list1 = read_video(v1)
    list2 = read_video(v2)

    for i, (im1,im2) in enumerate(zip(list1,list2)):

        print('step ', i)


        disparity = run_on_stereo(im1 , im2)    

        disparity = np.float16(disparity)

        disparity = cv2.cvtColor(disparity, cv2.COLOR_RGB2BGR)

        cv2.imwrite('disp' +str(i) +'.jpeg',disparity)


'''







'''
for R

 # kp1, des1 = extract_features(im1)
        # kp2, des2 = extract_features(im2)

        # # get matches between a pair of frames
        # _ , matches = match_features(des1, des2, filtration_threshold)

        # image1_points = []
        # image2_points = []

        # for m in mtch:
        #     m = m[0]
        #     query_idx = m.queryIdx
        #     train_idx = m.trainIdx

        #     # get first img matched keypoints
        #     p1_x, p1_y = kp1[query_idx].pt
        #     image2_points.append([p1_x, p1_y])

        #     # get second img matched keypoints
        #     p2_x, p2_y = kp2[train_idx].pt
        #     image1_points.append([p2_x, p2_y])

        # E, mask = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), K)
        # _, R, t, mask = cv2.recoverPose(E, np.array(image1_points), np.array(image2_points), K)
 

        # kernel_size = 7
        # im1 = cv2.GaussianBlur(im1, (kernel_size,kernel_size), 1.5)
        # im2 = cv2.GaussianBlur(im2, (kernel_size, kernel_size), 1.5)

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        # disparity = stereo.compute(im1,im2)
'''
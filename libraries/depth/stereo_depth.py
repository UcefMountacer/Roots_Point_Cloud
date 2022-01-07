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
    
    kernel_size = 7
    smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
    smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)
    

    left_matcher = cv2.StereoSGBM_create(
        numDisparities=16,  
        blockSize=5)
        
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 8000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(smooth_left, smooth_right)  
    dispr = right_matcher.compute(smooth_right, smooth_left)  

    filteredImg = wls_filter.filter(displ, smooth_left, disparity_map_right=dispr)  

    return filteredImg

def run_on_stereo(left , right, rectify , K):

    '''
    run code
    '''

    if rectify:

        K1, K2 = K, K

        height, width, channel = left.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, None, None, None, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, None, None, None, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disp = depth_map(gray_left, gray_right) 

    if not rectify:
            
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disp = depth_map(gray_left, gray_right) 

    return disp

if __name__ == '__main__':

    K = np.array([[915.4837025,0,971.17515329],
                  [0,578.96998153,540.72585068],
                  [0,0,1]])

    # directories
    dir1 = '/media/youssef/ubuntu_data/WORK/Roots/MVI_0252'
    dir2 = '/media/youssef/ubuntu_data/WORK/Roots/MVI_0590'
    rectify = 1
    
    output_dir = 'depth/output'
    # input data
    list1 = sorted(os.listdir(dir1))
    list2 = sorted(os.listdir(dir2))

    for i, (im1p,im2p) in enumerate(zip(list1,list2)):

        print('step ', i)

    
        im1 = cv2.imread(os.path.join(dir1,im1p),0)
        im2 = cv2.imread(os.path.join(dir2,im2p),0)

        # kernel_size = 7
        # im1 = cv2.GaussianBlur(im1, (kernel_size,kernel_size), 1.5)
        # im2 = cv2.GaussianBlur(im2, (kernel_size, kernel_size), 1.5)

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(im1,im2)

        cv2.imwrite(os.path.join(output_dir,'disp' +str(i) +'.png'),disparity)
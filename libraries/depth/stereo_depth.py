import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

###############
# stereo depth
###############



def load_params(path):

    '''
    Loads camera matrix and distortion coefficients
    '''

    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    cv_file.release()

    return camera_matrix


def init_stereo_method(minDisparity = 0 , numDisparities = 16*45, blockSize = 1, lmbda = 40000, sigma = 2):

    '''
    initialize stereo matchers nd filters
    '''

    stereo = cv2.StereoSGBM_create(
                        numDisparities=numDisparities,  
                        blockSize=blockSize,
                        minDisparity=minDisparity,
                        disp12MaxDiff=1,
                        uniquenessRatio=10,
                        speckleWindowSize=150,
                        speckleRange=32)

    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    return stereo, right_matcher, wls_filter, minDisparity, numDisparities


def scale_disparity(disp, methods):

    '''
    scale diqparity for opencv format
    '''

    _, _, _, minDisparity, numDisparities = methods

    
    disp = (disp - minDisparity) / numDisparities
    disp += abs(np.amin(disp))
    disp /= np.amax(disp)
    disp[disp < 0] = 0

    d = np.array(255 * disp, np.uint8) 
    d[d < 1] = 1

    return d


def scale_depth(depth, min, max):

    '''
    scale depth for opencv format
    '''

    depth[depth > max] = max
    depth[depth < min] = min

    depth_viz = (depth - min) / max
    depth_viz += abs(np.amin(depth_viz))
    depth_viz /= np.amax(depth_viz)
    depth_viz[depth_viz < 0] = 0

    depth_viz = np.array(255 * depth_viz, np.uint8) 

    return depth, depth_viz


def disp_2_depth(disp, b, f):

    '''
    depth = baseline * focal / disparity
    '''

    return b * f / disp


def run_on_stereo(left , right, methods, base, focal, max_depth= 500 , min_depth = 0.1):

    '''
    run stereo vision
    '''
            
    # convert to grascale before calculating depth

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # get functions from initializer
    stereo, right_matcher, wls_filter, _, _ = methods

    # compute stereo vision in respect to left and right image
    ''' The cutting problem occurs here '''
    displ = stereo.compute(left, right)  
    dispr = right_matcher.compute(right, left)  

    # apply filter using bth 
    disp = wls_filter.filter(displ, left, disparity_map_right=dispr)

    # solving opencv problem
    # using disparity to get depth without this scaling yields bad results
    disp = scale_disparity(disp , methods)

    # use scaled disparity to get depth
    depth = disp_2_depth(disp, base, focal)

    # scale depth for vizualisation and saving
    # pure depth has some inf values that need to be truncated
    # depth is to be used for 3d reconstruction
    # depth_viz is what will be shown using opencv (video)
    depth, depth_viz = scale_depth(depth, min_depth, max_depth)

    # for opencv visualization
    disp = cv2.cvtColor(disp,cv2.COLOR_GRAY2BGR)
    depth_viz = cv2.cvtColor(depth_viz,cv2.COLOR_GRAY2BGR)


    return depth, depth_viz, disp







# def generate_video(frames, output_dir):

#     '''
#     generate and save a video from a list of frames
#     '''
   
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     height, width= frames[0].shape[0], frames[0].shape[1]
#     video = cv2.VideoWriter(output_dir, fourcc, 20, (width,height))

#     for frame in frames:

#         video.write(frame)

#     cv2.destroyAllWindows()
#     video.release()

# def read_video(video_file_path):

#     list_of_frames = []
#     vidcap = cv2.VideoCapture(video_file_path)

#     success,image = vidcap.read()

#     while success:
#         list_of_frames.append(image)
#         success,image = vidcap.read()

#     return list_of_frames

# def adjust_gamma(image, gamma):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")
 
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

# def image_enhance(img):

#     # CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

#     # convert from BGR to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  
    
#     # split on 3 different channels
#     l, a, b = cv2.split(lab)  

#     # apply CLAHE to the L-channel
#     l2 = clahe.apply(l)  

#     # merge channels
#     lab = cv2.merge((l2,a,b))  
    
#     # convert from LAB to BGR
#     img_enhance = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  
    
#     return img_enhance

# def correct_gamma(image):
    
#     #get size of image
#     img_height, img_width = image.shape[:2]
    
#     # apply gamma correction and show the images
#     gamma = 1.5
#     gamma = gamma if gamma > 0 else 0.1
#     adjusted = adjust_gamma(image, gamma=gamma)
#     enhanced_image = image_enhance(adjusted)

#     return enhanced_image

    

# if __name__ == '__main__':


#     # directories
#     v1 = 'data/43/MVI_0252.MOV'
#     v2 = 'data/43/MVI_0590.MOV'
    
#     output_dir = 'libraries/depth/output'
#     # input data+
#     list1 = read_video(v1)[:10]
#     list2 = read_video(v2)[:10]

#     list1 = list(map(correct_gamma , list1))
#     list2 = list(map(correct_gamma , list2))

#     methods = init_stereo_method()

#     depth_list = []

#     for i, (im1,im2) in enumerate(zip(list1,list2)):

#         print('step ', i)

#         # m = mask(list1[i],list1[i+1])

#         depth, depth_viz, disp = run_on_stereo(im1 , im2, methods,base=100, focal=730, max_depth= 500 , min_depth = 0)    

#         # depth = segment_background(depth, m)

#         depth_list.append(depth_viz)

#         # if i == len(list1) - 2:
#         #     break
    
#     generate_video(depth_list, '/media/youssef/ubuntu_data/WORK/Roots_Point_Cloud/outputs/depth/depth_enhanced.MOV')







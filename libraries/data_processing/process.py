
import cv2
import os
import numpy as np
import skimage.measure
import matplotlib.patches as mpatches

def read_video(video_file_path):

    ''' 
    read video and return frames
    '''

    list_of_frames = []
    vidcap = cv2.VideoCapture(video_file_path)

    success,image = vidcap.read()

    while success:
        list_of_frames.append(image)
        success,image = vidcap.read()

    return list_of_frames


def generate_video(frames, output_dir):

    '''
    generate and save a video from a list of frames
    '''
   
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    height, width= frames[0].shape[0], frames[0].shape[1]
    video = cv2.VideoWriter(output_dir, fourcc, 20, (width,height))

    for frame in frames:

        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def segment_background(im):

    '''
    remove background using color segmentation
    '''
    
    lower = np.array([180, 48, 35])
    upper = np.array([245, 145, 128])
    thresh = cv2.inRange(im, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = 255 - morph
    result = cv2.bitwise_and(im, im, mask=mask)

    return result


def adjust_gamma(image, gamma):

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def image_enhance(img):

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  
    
    # split on 3 different channels
    l, a, b = cv2.split(lab)  

    # apply CLAHE to the L-channel
    l2 = clahe.apply(l)  

    # merge channels
    lab = cv2.merge((l2,a,b))  
    
    # convert from LAB to BGR
    img_enhance = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  
    
    return img_enhance


def correct_gamma(image):
    
    # apply gamma correction and show the images
    gamma = 2
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(image, gamma=gamma)
    enhanced_image = image_enhance(adjusted)

    return enhanced_image


def remove_blur(im):

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(im, -1, sharpen_kernel)

    return sharpen


def mask_root(image):

    cropped_image = image[0:580, 700:1200]
   
    # switch to HSV color space, better to pick intervals than RGB
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # specify the lower and upper boundaries for the color segmentation
    low= np.array([0, 129, 173])
    high = np.array([121, 255, 255])

    # take the foreground with the specified boundaries
    mask = cv2.inRange(hsv, low, high)

    # label = skimage.measure.label(mask)
    # prop = skimage.measure.regionprops(label)            
    
    # apply it on the original frame and get its original form
    res = cv2.bitwise_not(cropped_image,cropped_image, mask= mask)

    return mask

if __name__ == '__main__':

    # directories
    v1 = 'data/43/MVI_0252.MOV'
    v2 = 'data/43/MVI_0590.MOV'
    
    output_dir = 'libraries/depth/output'
    # input data+
    list1 = read_video(v1)
    list2 = read_video(v2)

    l = []

    for i, (im1,im2) in enumerate(zip(list1,list2)):

        # im1 = remove_blur(correct_gamma(im1))
        mask = mask_root(im1)

        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        l.append(mask)

    
    generate_video(l, '/media/youssef/ubuntu_data/mask.MOV')






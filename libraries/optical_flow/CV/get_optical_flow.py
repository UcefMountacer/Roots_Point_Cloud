import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def convertToOptical(prev_image, curr_image):

    '''
    return flow from 2 consecutive images
    '''

    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.25, 3, 20, 3, 5, 1.2, 0)
    # flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)

    hsv = np.zeros_like(prev_image)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_image_bgr


''' region of interest '''
polygon = np.array( [[[400,0],[1500,0],[1500,500], [400,500]]], dtype=np.int32 )

def region_of_interest(img , polygon=polygon):

    '''
    return region of interest based on a polygon of pixels
    '''

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image








import cv2
import os
import numpy as np

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






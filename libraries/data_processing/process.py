
import numpy as np
import cv2
import os

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
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(os.path.join(output_dir , 'optical_flow.MOV'), fourcc, 20, (width,height))

    for frame in frames:

        video.write(frame)

    cv2.destroyAllWindows()
    video.release()






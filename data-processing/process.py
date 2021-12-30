import cv2
import os
import numpy as np

def convert_video_to_frames(path_video , FPS, output_images_path):

    '''
    convert video to frames with estimated FPS
    '''

    vidcap = cv2.VideoCapture(path_video)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(output_images_path,"image"+str(count)+".jpg"), image)     # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 1/FPS
    count = 1000
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 4)
        success = getFrame(sec)
        print(count)





''' RUN '''

if __name__ == '__main__':

    video = 'MVI_0252.MOV'
    FPS = 10
    output_dir = 'MVI_0252'
    convert_video_to_frames(video,FPS,output_dir)

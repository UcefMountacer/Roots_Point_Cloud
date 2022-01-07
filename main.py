

import os
import subprocess
from argparse import ArgumentParser
from libraries.data_processing.process import *
from libraries.optical_flow.CV.get_optical_flow import *
from libraries.auto_calibration.calibration_ORB_features import *
from libraries.optical_flow.RAFT.run import *


description = """ Run optical flow from a video and generate output video"""

def parse_args():

    parser = ArgumentParser(description=description)

    #specify operation to do
    parser.add_argument('--op', type=str, help="Opertaion to execute: 'OF' for optical flow, \
                                                'C' for auto calibration , \
                                                'OF_R' for optical flow using raft model, \
                                                'D' for depth" , required=1)

    #if optical flow
    parser.add_argument('--video', type=str, help="Path to video of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--output_video' , type=str, help="Path to save output video of optical flow", default='outputs/optical_flow')

    #if auto calibration
    parser.add_argument('--v1', type=str, help="Path to video 1 of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--v2' , type=str, help="Path to video 2 of the root", default='data/MVI_0590.MOV')
    parser.add_argument('--filter_th', type=float, help = "lowe distance between matches threshold", default=0.9)
    parser.add_argument('--rms_th', type=float, help = "root mean square threshold to discard pairs with bad K", default=10.0)
    parser.add_argument('--save_K', default = 'data/calibration.yml')

    #if optical flow using RAFT
    parser.add_argument('--video_r', type=str, help="Path to video of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--output_video_raft' , type=str, help="Path to save output video of optical flow", default='outputs/RAFT')
    
    parser.add_argument('--model', type=str, help="Path to save downloaded model weights", default='libraries/optical_flow/RAFT/models/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model', required=0)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', required=0)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation', required=0)
    
    # if depth
    parser.add_argument('--output_video_depth' , type=str, help="Path to save output video of optical flow", default='outputs/depth')

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    # get type of operation to do
    operation = args.op

    if operation == 'OF':

        # optical flow

        print('running optical flow using opencv')

        if not os.path.exists(args.output_video):

            os.mkdir(args.output_video)
            print('output file created')
    
        # get paths from argparse
        video_path = args.video
        output_flow_dir = args.output_video

        # extract frames from video
        list_of_frames = read_video(video_path)

        #define a list to store optical flow arrays
        list_of_outputs = []

        for i, (im1, im2) in enumerate(zip(list_of_frames[:-1], list_of_frames[1:])):

            print('processing for pair number :',i)

            # get optical flow for the whole frame
            flow = convertToOptical(im1,im2)

            # take only the region that is interesting to nalyse
            flow = region_of_interest(flow)

            #add array to list of outputs
            list_of_outputs.append(flow)

        # generate and save video from arrays of optical flow
        generate_video(list_of_outputs, os.path.join(output_flow_dir , 'optical_flow.MOV')) 

    if operation == 'C':

        # auto-calibration

        print('running auto-calibration')

        video1 = args.v1
        video2 = args.v2

        # lowe distance between matches threshold
        filtration_threshold = args.filter_th

        # rms error of calibration threshold
        rms_threshold = args.rms_th
        
        # input frames from videos
        list1 = read_video(video1)
        list2 = read_video(video2)

        print('running auto-calibration using',len(list1), 'images')

        # init an empty K
        K = np.zeros((3,3))

        # define a list to store Ks with low RMS error
        k_list = []
        err_list = []


        for i, (im1,im2) in enumerate(zip(list1,list2)):

            print('step :', i)

            # get ORB features and their descriptors
            kp1, des1 = extract_features(im1)
            kp2, des2 = extract_features(im2)

            # get matches between a pair of frames
            _ , matches = match_features(des1, des2, filtration_threshold)

            # calibration data operation : needed for opencv camera calibration process in this way

            pattern_keys = list(kp1)
            frames_keys=[list(kp1),list(kp2)]

            frames_matches = [matches]

            img_size = (im1.shape[1], im1.shape[0])

            [obj_pts, image_pts] = to_calibration_data(frames_matches, pattern_keys, frames_keys, 2)


            # get K matrix and error
            K, err = calibrate_intrinsic(obj_pts, image_pts, img_size)

            print('pair number :', i, 'with error :',err)

            # append to list of Ks and list or respective errors
            err_list.append(err)
            k_list.append(K, args.save_K)


        # define a list of good indexes (with good rms error)
        good_calib_indexes = []
        for i,err in enumerate(err_list):

            if err < rms_threshold:
                good_calib_indexes.append(i)


        # get mean of all good Ks
        K1 = mean_k(k_list, good_calib_indexes)

        # get K with minimal error
        K2 = k_list[np.argmin(err_list)]

        print('K obtained with calculating mean of K with low RMS error: ',K1)

        print('K with the lowest RMS error :',K2)

        save_K(K, )

    if operation == 'OF_R':



        model_path = args.model

        if not os.path.exists(model_path):

            #model not downloaded, should be done
            print('downloading model')

            subprocess.call('./libraries/optical_flow/RAFT/download_models.sh')

        if not os.path.exists(args.output_video_raft):

            os.mkdir(args.output_video_raft)

        else:

            # optical flow using RAFT

            print('running optical flow using RAFT model')
        
            # get paths from argparse
            video_path = args.video
            output_flow_dir = args.output_video_raft

            # create output folder if not exists

            # extract frames from video
            list_of_frames = read_video(video_path)

            #define a list to store optical flow arrays
            list_of_outputs = []

            demo(args, list_of_frames, list_of_outputs)
            
            generate_video(list_of_outputs, os.path.join(output_flow_dir , 'optical_raft.MOV'))

    if operation == 'D':

        if not os.path.exists(args.output_video_depth):

            os.mkdir(args.output_video_depth)
            print('output directory created')

        print('running disparity')
        
        output_dir = args.output_video_depth
        # input data
        list1 = read_video(args.v1)
        list2 = read_video(args.v2)

        depth_list = []

        for i, (im1,im2) in enumerate(zip(list1,list2)):

            print('pair number :', i)

            disp = np.zeros(im1.shape)

            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(im1,im2)

            disparity = np.array(disparity, dtype=np.uint8)

            d = cv2.cvtColor(disparity,cv2.COLOR_GRAY2BGR)

            depth_list.append(d)


        generate_video(depth_list, os.path.join(output_dir , 'disparity.MOV')) 

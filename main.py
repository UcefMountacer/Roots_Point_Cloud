

import os
import subprocess
from argparse import ArgumentParser
from libraries.data_processing.process import *
from libraries.optical_flow.CV.get_optical_flow import *
from libraries.auto_calibration.calibration_ORB_features import *
from libraries.optical_flow.RAFT.run import *
from libraries.depth.stereo_depth import *


description = """ Run optical flow from a video and generate output video"""

def parse_args():

    parser = ArgumentParser(description=description)

    #specify operation to do
    parser.add_argument('--op', type=str, help="Opertaion to execute: 'OF' for optical flow, \
                                                'C' for auto calibration , \
                                                'D' for depth" , required=1)

    #if optical flow
    parser.add_argument('--video', type=str, help="Path to video of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--output_video' , type=str, help="Path to save output video of optical flow", default='outputs/optical_flow')

    #if auto calibration
    parser.add_argument('--v1', type=str, help="Path to video 1 of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--v2' , type=str, help="Path to video 2 of the root", default='data/MVI_0590.MOV')
    parser.add_argument('--filter_th', type=float, help = "lowe distance between matches threshold", default=0.9)
    parser.add_argument('--rms_th', type=float, help = "root mean square threshold to discard pairs with bad K", default=10.0)
    parser.add_argument('--save_K', default = 'outputs')

    # if depth
    parser.add_argument('--output_video_depth' , type=str, help="Path to save output video of optical flow", default='outputs/depth')
    parser.add_argument('--baseline' , type=float, help="baseline of stereo rig in mm", default=100)
    parser.add_argument('--opd' , type=bool, help="save npy (0) or visualize disparity in a video (1)", default=1)

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
            k_list.append(K)


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

        if args.save_K:
            
            save_path = os.path.join(args.save_K , 'calibration_ORB.yml')
            save_K(K, save_path)

    if operation == 'D':

        if not os.path.exists(args.output_video_depth):

            os.mkdir(args.output_video_depth)
            print('output directory created')

        print('running disparity and depth')
        
        output_dir = args.output_video_depth
        # input data
        list1 = read_video(args.v1)[:10]
        list2 = read_video(args.v2)[:10]

        images_list = []

        # initialize methods

        methods = init_stereo_method()

        # operation

        op_depth = args.opd

        if op_depth == 1:

            # show disparity

            for i, (im1,im2) in enumerate(zip(list1,list2)):

                print('stereo pair number :', i)

                depth, depth_viz, disp = run_on_stereo(im1 , im2, methods,base=100, focal=730, max_depth= 500 , min_depth = 0)

                disp = cv2.cvtColor(disp,cv2.COLOR_GRAY2BGR)
                depth_viz = cv2.cvtColor(depth_viz,cv2.COLOR_GRAY2BGR)

                cobined_image = np.hstack((disp, depth_viz))

                images_list.append(cobined_image)

            generate_video(images_list, os.path.join(output_dir , 'combined.MOV')) 

        if op_depth == 0:

            # save depth (TO DO)

            K = load_params(path= 'outputs/calibration.yml')

            fx = K[0][0]
            b = args.baseline
            parameters = fx, b

            pass

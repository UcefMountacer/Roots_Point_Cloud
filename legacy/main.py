

import os
import subprocess
from argparse import ArgumentParser
from libraries.data_processing.process import *
from legacy.libraries.optical_flow.RAFT.run import *


description = """ Run optical flow from a video and generate output video"""

def parse_args():

    parser = ArgumentParser(description=description)


    #if optical flow using RAFT
    parser.add_argument('--video_r', type=str, help="Path to video of the root", default='data/MVI_0252.MOV')
    parser.add_argument('--output_video_raft' , type=str, help="Path to save output video of optical flow", default='outputs/RAFT')
    
    parser.add_argument('--model', type=str, help="Path to save downloaded model weights", default='libraries/optical_flow/RAFT/models/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model', required=0)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', required=0)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation', required=0)
    

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    # get type of operation to do

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

    
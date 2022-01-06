

from libraries.data_processing.process import *
from libraries.optical_flow.CV.get_optical_flow import *




if __name__ == "__main__":

    video_path = 'data/MVI_0252.MOV'

    output_flow_dir = 'outputs/optical_flow'

    list_of_frames = read_video(video_path)

    for i, (im1, im2) in enumerate(zip(list_of_frames[:-1], list_of_frames[1:])):

        flow = convertToOptical(im1,im2)
        flow = region_of_interest(flow)
        
        plt.imsave(os.path.join(output_flow_dir,'optical_flow_' +str(i) +'_.png'),flow)


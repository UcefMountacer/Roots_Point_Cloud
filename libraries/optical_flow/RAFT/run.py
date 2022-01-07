import sys
sys.path.append('libraries/optical_flow/RAFT/core')

import numpy as np
import torch

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'


def demo(args, images, list_of_outputs):

    '''
    run demo of raft model
    args contain infos : model, small, mixed_precision, alternate_corr

    return list of output
    
    '''
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        for i, (image1, image2) in enumerate(zip(images[:-1], images[1:])):

            print('pair number :',i)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            img = img[0].permute(1,2,0).cpu().numpy()
            flo = flo[0].permute(1,2,0).cpu().numpy()
            
            # map flow to rgb image
            flo = flow_viz.flow_to_image(flo)
            img_flo = np.concatenate([img, flo], axis=0)

            final = img_flo / 255.0

            list_of_outputs.append(final)

    return list_of_outputs






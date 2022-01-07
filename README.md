# Roots_Point_Cloud


## install packages for conda (work in progress)

Run these commands, starting with the first one that will create an venv using python 3.8 that works for opencv 4
```
conda create -n newenv -c conda-forge python=3.8 opencv
conda install -c conda-forge scipy
conda install -c anaconda setuptools=45.2.0
conda install -c conda-forge numpy=1.19.5
conda install -c conda-forge pillow=8.3.2
conda install -c conda-forge matplotlib==3.4.2
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## steps to run these fonctions (work in progress)

### optical flow

```
python3 main.py --op 'OF' --video path_to_video --output_video path_to_save_optical_flow_video
```

for example:

```
python3 main.py --op 'OF' --video data/MVI_0252.MOV --output_video outputs/optical_flow
```


### auto-calibration

```
python3 main.py --op C --v1 path_to_video1 --v2 path_to_video2 --filter_th TH1 --rms_th TH2
```

TH1 is from 0 to 1 (0.9 as an example)
TH2 is a float, to be adjusted later when we have an idea of ground truth camera parameters

for example:

```
python3 main.py --op C --v1 data/MVI_0252.MOV --v2 data/MVI_0590.MOV --filter_th 0.9 --rms_th 10.0
```

### RAFT for optical flow

first, you need to download pre-trained models

```
python3 main.py --op OF_R
```

Then run the command to run the model (not tested locally)

```
python3 main.py --op OF_R --video_r data/MVI_0252.MOV --output_video_r outputs/RAFT --model libraries/optical_flow/RAFT/models/raft-things.pth
```

### depth map (disparity for now)

```
python3 main.py --op D --v1 data/MVI_0252.MOV --v2 data/MVI_0590.MOV --output_video_depth outputs/depth
```
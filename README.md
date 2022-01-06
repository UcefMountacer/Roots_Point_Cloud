# Roots_Point_Cloud


## install packages for conda (work in progress)


`conda install matplotlib`

`conda install numpy`

`conda install -c menpo opencv`

`conda install pillow`

`conda install scipy`

`conda install -c anaconda setuptools`

this command is problematic for your case

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`



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
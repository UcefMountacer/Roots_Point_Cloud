# Roots_Point_Cloud


## install packages using conda

`conda install --file requirements.txt`

## steps to run these fonctions (work in progress)

### **0- extract frames from video**

go to data-processing/process.py

supply these informations in the main function:

```
# video 1
video = 'data/MVI_0252.MOV'
FPS = 10
output_dir = 'data/MVI_0252'
convert_video_to_frames(video,FPS,output_dir)

# video 2
video = 'data/MVI_0590.MOV'
FPS = 10
output_dir = 'data/MVI_0590'
convert_video_to_frames(video,FPS,output_dir)
```

### **to generate optical flow**

go to optical-flow/get_optical_flow.py

supply these informations in the main function:

```
image_dir = 'data/MVI_0590'
output_dir = 'optical-flow/output'
```

### **auto calibration using ORB features**

go to auto-calibration/calibration_ORB_features.py

suppmy these informations in the main:

```
# directories
dir1 = 'MVI_0590/'
dir2 = 'MVI_0252/'
# lowe distance between matches threshold
filtration_threshold = 0.9
# rms error of calibration threshold
rms_threshold = 10
```


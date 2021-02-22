# BRIEF Descripton
A pipeline to detect anomaly action in a video.

## MMPOSE FOR POSE ESTIMATION

The official github is: https://github.com/open-mmlab/mmpose.

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

TODO:
Download pretrained mmpose models from https://drive.google.com/drive/folders/1DuPVcAaaTr-sjz-0fTiY76EUMTRaxJfU?usp=sharing and put it in directory /speed-up-mmpose/pretrained/.

## MOT MODEL FOR HUMAN TRACKING

The official github is: https://github.com/Zhongdao/Towards-Realtime-MOT.

TODO:
Download pretrained tracking models from https://github.com/Zhongdao/Towards-Realtime-MOT, there are three of them. Download the neccessary model weight and put it in /tracking/Towards-Realtine-MOT/weights/. Because these files are large, we suggest adding it to your drive and then use !cp command in colab instead of downloading.
This github is using JDE-576x320. If you wish to change the model, please modify file skeleton_detect.py.

*** Most of the modifications are made on main.py, skeleton_detect.py, dataset.py in speed-up-mmpose/.

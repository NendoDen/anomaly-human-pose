# BRIEF Descripton
A pipeline to detect anomaly action in a video.

*** Most of the modifications are made on main.py, skeleton_detect.py, dataset.py in speed-up-mmpose/.

## MOT MODEL FOR HUMAN TRACKING

The official github is: https://github.com/Zhongdao/Towards-Realtime-MOT.

TODO:
Download pretrained tracking models from https://github.com/Zhongdao/Towards-Realtime-MOT, there are three of them. Download the neccessary model weight and put it in /tracking/Towards-Realtine-MOT/weights/. Because these files are large, we suggest adding it to your drive and then use !cp command in colab instead of downloading.
This github is using JDE-576x320. If you wish to change the model, please modify line 48 in skeleton_detect.py.

## MMPOSE FOR POSE ESTIMATION

The official github is: https://github.com/open-mmlab/mmpose.

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

## QUICK START BY USING COLAB

https://colab.research.google.com/drive/1pTH-d2ph0ad3or8bZ1d0Hb5KHAAmQ8RB?usp=sharing#scrollTo=RbTrYHLt8ntR

## SOME OTHER MODIFICATION TIPS:

The current tracking model is used in order to reduce the tracking time. In the past, we used mmtracking for tracking. If use want to switch to mmtracking, comment line 89-92 and use line 83-84 instead. 

The current issues are:

1/ The tracking model cannot detect abnormal skeleton (when people fall down or jump).
2/ The anomaly detection part is not fully handled.

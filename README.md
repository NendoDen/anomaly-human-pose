# anomaly-human-pose
A pipeline to detect anomaly action in a video.

Download pretrained mmpose models from https://drive.google.com/drive/folders/1DuPVcAaaTr-sjz-0fTiY76EUMTRaxJfU?usp=sharing and put it in directory speed-up-mmpose/pretrained/.

Download pretrained tracking models from https://github.com/Zhongdao/Towards-Realtime-MOT, there are three of them. This github is using JDE-576x320. If you wish to change the model, please modify file skeleton_detect.py.

*** Most of the modifications are made on main.py, skeleton_detect.py, dataset.py in speed-up-mmpose/.

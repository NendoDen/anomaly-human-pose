import os
import cv2
import time
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_tracking_result)

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import init_model as init_tracking_model
    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False

class SkeletonTracking:
    def __init__(self, detector_config, tracker_config, output_path, det_path):
        self.tracking_model = init_tracking_model(tracker_config, None, device='cuda:0')
        self.pose_model = init_pose_model(detector_config, 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth', device='cuda:0')
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.tracking_result = {}
        self.output_path = output_path
        self.det_path = det_path  

    def process_mmtracking_results(self, mmtracking_results):
        """Process mmtracking results.

        :param mmtracking_results:
        :return: a list of tracked bounding boxes
        """
        person_results = []
        for track in mmtracking_results['track_results'][0]:
            person = {}
            person['track_id'] = int(track[0])
            person['bbox'] = track[1:]
            # print(person['bbox'])
            person_results.append(person)
        return person_results

    def get_tracking_result(self, file_path, cap):
        result = {}
        f = open(file_path, 'r')  
        width = cap.get(3)
        height = cap.get(4)

        for line in f:
            frame_id, track_id, x1, y1, w, h, _, _, _, _ = [float(i) for i in line.split(',')]
            x1, y1, w, h = int(x1*width/576), int(y1*height/320), int(w*width/576), int(h*height/320)
            if frame_id not in result:
                result[int(frame_id)] = []
            new_result = {}
            new_result['track_id'] = int(track_id)-1
            new_result['bbox'] = np.array([x1, y1, x1+w, y1+h, 0.9])
            
            result[int(frame_id)].append(new_result)
        return result

    def getSkeleton(self, cap):
        
        very_start_time = time.time()

        frame_id = 0
        infer_mot_time = 0
        tracking_time = 0
        posing_time = 0
        prepair_inp_time = 0
        result = {}
        video_frames = []

        persons_from_bach = self.get_tracking_result(self.det_path, cap)


        while (cap.isOpened()):
            start_time = time.time()

            flag, img = cap.read()
            if not flag:
                break

            # print("IMAGE SIZE = ", img.shape)


            # mmtracking_results = inference_mot(self.tracking_model, img, frame_id=frame_id)
            # person_results = self.process_mmtracking_results(mmtracking_results)

            # infer_mot_time += time.time()-start_time
            # start_time = time.time()

            if int(frame_id) in persons_from_bach:
                person_results = persons_from_bach[int(frame_id)]
            else:
                person_results = []
            # print("PERSON RESULT")
            # print(person_results)

            tracking_time += time.time() - start_time
            start_time = time.time()

            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results,
                bbox_thr=0.3,
                format='xyxy',
                dataset=self.dataset,
                return_heatmap=False,
                outputs=None)

            posing_time += time.time() - start_time
            start_time = time.time()
            # print('===============POSE RESULTS===============')
            # print(pose_results)
            for ske in pose_results:
                # print("Skeleton ==========><==========")
                # print(ske)

                name_of_ske = ''
                track_id = ske['track_id']
                arr_coor = ske['keypoints']
                if track_id < 10:
                    name_of_ske = '0001_000' + str(track_id)
                elif track_id < 100:
                    name_of_ske = '0001_00' + str(track_id)
                elif track_id < 1000:
                    name_of_ske = '0001_0' + str(track_id)
                else: name_of_ske = '0001_' + str(track_id)

                format_coor = []
                for coor in arr_coor:
                  format_coor.append(coor[0])
                  format_coor.append(coor[1])

                if name_of_ske in result:
                    result[name_of_ske]['frames'].append(frame_id)
                    result[name_of_ske]['coordinates'].append(format_coor)
                else:
                    result[name_of_ske] = {}
                    result[name_of_ske]['frames'] = [frame_id]
                    result[name_of_ske]['coordinates'] = [format_coor]

            frame_id += 1

            if int(frame_id) in persons_from_bach:
                width = cap.get(3)
                height = cap.get(4)
                for person in persons_from_bach[int(frame_id)]:
                    x1, y1, x2, y2, _ = person['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    R_id = person['track_id']*50%256 
                    G_id = person['track_id']*25%256 
                    B_id = person['track_id']*10%256 
                    img = cv2.rectangle(img, (x1,y1), (x2, y2), thickness = 2,\
                                        color=(R_id, G_id, B_id))
                for ske in pose_results:
                    for kp in ske['keypoints']:
                        if kp[2] > 0.5:
                            img = cv2.circle(img, (kp[0],kp[1]), radius=10, color=(0, 255, 0), thickness=-1)

            video_frames.append(img)

            prepair_inp_time += time.time() - start_time
            start_time = time.time()

        
        print("NUMBER OF FRAME: ", frame_id)
        print("Inference time = ", infer_mot_time)
        print("Tracking time = ", tracking_time)
        print('Posing time = ', posing_time)
        print('Prepair inp time = ', prepair_inp_time)
        print("TOTAL TIME: ", time.time()-very_start_time)

        return frame_id+1, result, video_frames
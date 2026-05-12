import os
import json
from utils import write_jsonl_file
import cv2
import random
from clevrer.utils import *


class CLEVRER_Dataset:
    def __init__(self, root_path, split, sample_num=200):
        self.root_path = root_path
        self.split = split
        self.ann_path = os.path.join(self.root_path, f'annotation_{split}')
        self.video_path = os.path.join(self.root_path, f'video_{split}')
        self.data_list, self.collision_dict = self._process_and_save_ann(sample_num)
    
    def _process_and_save_ann(self, sample_num):
        ann_list = []
        collision_dict = dict()
        ann_path_list = os.listdir(self.ann_path)
        
        for idx, ann_file in enumerate(ann_path_list):
            
            if '.json' not in ann_file:
                continue
            ann_name = ann_file.split('.json')[0].split('_')[-1]
            ann_list.append({ann_name: self._process_single_ann(ann_file)})
            len_col = len(ann_list[-1][ann_name]['collision'])
            collision_list = collision_dict.get(len_col, list())
            collision_list.append((idx, ann_name))
            collision_dict[len_col] = collision_list.copy()
            
            if idx == sample_num - 1:
                break
            
        return ann_list, collision_dict
    
    def _process_single_ann(self, ann_file):
        with open(os.path.join(self.ann_path, ann_file)) as file:
            raw_ann_dict = json.load(file)
        
        
        return raw_ann_dict
        
    def _process_video(self, video_file):
        import cv2
        cap = cv2.VideoCapture(video_file)
    
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frames = []

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames
    
    def _get_velocity_item(self, idx):
        (ann_name, ann_dict), = self.data_list[idx].items()
        video_path = os.path.join(self.video_path, f"video_{ann_name}.mp4")
        
        max_velocity, fastest_object, frame_idx = get_fastest_speed(ann_dict)
        return video_path, max_velocity, fastest_object, frame_idx
    

    def _get_collision_item(self, idx):
        (ann_name, ann_dict), = self.data_list[idx].items()
        video_path = os.path.join(self.video_path, f"video_{ann_name}.mp4")
        collision_list = ann_dict['collision']
        return video_path, collision_list
    
    def __len__(self):
        return len(self.data_list)
    
    



    

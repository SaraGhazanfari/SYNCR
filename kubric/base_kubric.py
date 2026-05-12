import os
import random
import numpy as np
from utils import read_json_file, write_jsonl_file


class KubricDataset:
    """Handles data loading and querying for Kubric scenes. No QA logic."""
    def __init__(self, path, view_count=3):
        self.path = path
        self.view_count = view_count   
        self.instance_data = dict()
        
    def get_raw_item(self, scene_idx: str):
        if scene_idx in self.instance_data:
            return self.instance_data[scene_idx]
        
        data_dict = dict()
        scene_path = os.path.join(self.path, scene_idx)
        
        for i in range(self.view_count):
            metadata_path = os.path.join(scene_path, f"cam{i}", "metadata.json")
            data_dict[f"cam{i}"] = read_json_file(metadata_path)['instances']
            
        self.instance_data[scene_idx] = data_dict
        return data_dict
    
    def count_object_occurrences(self, scene_idx, all_objects_in_scene):
        name_counts = {}
        for obj in all_objects_in_scene:
            base_name = self.get_object_name(scene_idx, obj)
            name_counts[base_name] = name_counts.get(base_name, 0) + 1
        return name_counts
        
    def get_visibility_for_all_objects(self, scene_idx, view_count):
        if scene_idx not in self.instance_data:
            self.get_raw_item(scene_idx)
            
        cam0_instances = self.instance_data[scene_idx]['cam0']
        object_count = len(cam0_instances)
        frame_count = len(cam0_instances[0]['visibility'])
        
        cam0_id_map = {obj['asset_id']: idx for idx, obj in enumerate(cam0_instances) if obj['asset_id'] is not None}
        visibility = np.zeros((view_count, object_count, frame_count))
        
        for view_id in range(view_count):
            cam_name = f"cam{view_id}" 
            for object_dict in self.instance_data[scene_idx][cam_name]:
                unique_id = object_dict['asset_id']
                if unique_id in cam0_id_map:
                    mapped_idx = cam0_id_map[unique_id]
                    visibility[view_id, mapped_idx, :] = object_dict['visibility']
                    
        return visibility

    def get_hit_frames(self, scene_idx, all_unique_objects):
        hit_frames = {}
        for obj_idx in all_unique_objects:
            positions = np.array(self.instance_data[scene_idx]['cam0'][obj_idx]['positions'])
            z_coords = positions[:, 2] 
            z_velocity = np.diff(z_coords)
            impacts = np.where((z_velocity[:-1] < 0) & (z_velocity[1:] >= 0))[0]
            hit_frames[obj_idx] = impacts[0] + 1 if len(impacts) > 0 else None
        return hit_frames

    def get_objects_visible_in_all_views(self, scene_idx, view_count):
        visibility = self.get_visibility_for_all_objects(scene_idx, view_count=view_count)
        visible_in_all_views = np.all(visibility > 0, axis=0) 
        all_unique_objects = set(np.where(visible_in_all_views.any(axis=1))[0])
        return all_unique_objects
    
    def get_object_name(self, scene_idx: str, object_idx: int) -> str:
        if scene_idx not in self.instance_data:
            self.get_raw_item(scene_idx)
        
        object_dict = self.instance_data[scene_idx]['cam0'][object_idx]
        shape = str(object_dict.get('asset_id', 'object')).split('_')[-1] 
        material = str(object_dict.get('material', '')).split('_')[-1]
        
        color = object_dict.get('color_label', '')
        if isinstance(color, (list, np.ndarray)): color = "" 
            
        size = object_dict.get('size', object_dict.get('scale', ''))
        if isinstance(size, (int, float)):
            size = "small" if size < 1.0 else "large"
        elif isinstance(size, (list, np.ndarray)):
            size = "small" if size[0] < 1.0 else "large"
            
        descriptors = [str(d) for d in [size, color, material, shape] if d]
        return " ".join(descriptors).strip().title()
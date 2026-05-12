import os
import numpy as np
from pathlib import Path
from utils import read_json_file
import cv2
import shutil


objects_to_filter = {
    
    'glass', 'cup', 'plate', 'bowl', 'decorative bowl', 'bowl of sweets', 'bowl of fruit', 
    'bottle', 'vase', 'flower vase', 'flowerpot', 'flower', 'candle', 'toy', 'ornament', 
    'hook', 'remote control', 'computer mouse', 'keyboard', 'device', 'tablet', 'sensor', 
    'fire alarm', 'paper', 'note', 'book', 'magazine', 'newspaper', 'binder', 'door knob', 
    'door hinge', 'self-closing mechanism', 'faucet', 'tap', 'clothes hanger', 'knife holder', 
    'ceiling light fixture connection', 'shoe', 'hat', 'bag', 'backpack',
    
    
    'clothes', 'blanket', 'throw blanket', 'towel', 'pillow', 'cushion', 'clutter', 
    'shelf with clutter', 'kitchen countertop item', 'decoration', 'kitchen decoration', 
    'wall hanging decoration', 'unknown', 'box', 'tray', 'pot', 'appliance', 'kitchen appliance',
    'kitchen counter', 
    
    
    'beam', 'support beam', 'pipe', 'wall panel', 'paneling', 'ceiling dome', 'ceiling duct', 
    'vent', 'air vent', 'ceiling vent', 'parapet', 'stairs', 'step', 'railing', 'handrail', 
    'stairs railing', 'banister', 'balustrade', 'curtain', 'window curtain', 'shower curtain', 
    'window shutter', 'window shutters', 'blinds', 'shades', 'window valence', 'curtain rod', 
    'curtain rail', 'curtain rod cover', 'frame', 'art frame', 'painting frame', 'photo mount', 
    'door frame', 'window frame', 'door/window frame', 'window glass', 'carpet', 'floor mat', 
    'doormat', 'rug', 'door/window', 'window/door', 'door window', 'wall electronics', 
    'drawer', 'kitchen cabinet door', 'cabinet', 'sliding door', 'wardrobe', 'window shade'
}



def classify_rooms_by_objects(object_list):
    """
    Parses a block of HM3D object strings and predicts the room type for each Region ID.
    """
    room_heuristics = {
        "Bathroom": ["toilet", "shower curtain", "washbasin", "bathtub"],
        "Bedroom": ["bed", "nightstand", "wardrobe"],
        "Kitchen": ["refrigerator", "freezer", "oven", "stove", "microwave", "kitchen cabinet"],
        "Living Room": ["couch", "sofa", "television", "tv", "fireplace"],
        "Utility/Garage": ["machine", "hose", "pipe", "broom", "heater", "bottle of detergent"],
        "Stairs/Hallway": ["staircase", "stairs", "doormat", "staircase handrail"]
    }
    
    scores = {room: 0 for room in room_heuristics}
    
    for obj in object_list:
        obj_lower = obj.lower()
        for room, anchors in room_heuristics.items():
            if any(anchor in obj_lower for anchor in anchors):
                scores[room] += 1
                
    best_match = max(scores, key=scores.get)
        
    if scores[best_match] == 0:
        best_match = "Generic/Unknown Room"
        scores[best_match] = 1
        
    return {
        "prediction": best_match,
        "score": scores[best_match],
        "sample_objects": object_list[:5]
    }
    
    
def get_frame_ranges(frame_list, max_gap=1):
    ranges = []
    if not frame_list:
        return ranges

    start = frame_list[0]
    prev = frame_list[0]

    for frame in frame_list[1:]:
        if frame <= prev + max_gap:
            prev = frame
        else:
            ranges.append((start, prev))
            start = frame
            prev = frame

    ranges.append((start, prev))
    return ranges


class Habitat_Dataset:
    def __init__(self, root_path, section, split):
        self.data_path = os.path.join(root_path, section, split)
        self.data_type = section
        self.root_path = root_path
        self.split = split 
        
        
        
    def _get_scene_path(self, scene_idx: str, route_idx: str = None):
        scene_path = os.path.join(self.data_path, scene_idx)
        if route_idx:
            scene_path = os.path.join(scene_path, route_idx)
        return scene_path

    def _get_semantic_path(self, scene_idx: str, route_idx: str = None):
        return os.path.join(self._get_scene_path(scene_idx, route_idx),
                            'scene_semantic_objects.json')

    def get_scene_object_info(self, scene_idx: str, route_idx: str = None, verbose: bool = False) -> tuple[dict, dict]:
        
        
        
        
        
        
        
                  
        semantic_path = self._get_semantic_path(scene_idx, route_idx)
        raw_object_info_dict = read_json_file(semantic_path)
        processed_object_info_dict = dict()
        aggregated_object_count_dict = dict()
        
        for key, value in raw_object_info_dict.items():
            if key == "0" or ' ' not in value: 
                continue
         
            
            value_parts = value.split(' ')
            object_name = ' '.join(value_parts[1:-1])
            
            if object_name in ["floor", "ceiling", "wall", "objects"]:
                continue
                
            aggregated_object_count_dict[object_name] = aggregated_object_count_dict.get(object_name, 0) + 1
            
            
            value_parts[0] = str(aggregated_object_count_dict[object_name])
            processed_object_info_dict[int(key)] = ' '.join(value_parts)
        
        if verbose:
            print(f"Scene {scene_idx} Route {route_idx} objects:")
            for name, count in aggregated_object_count_dict.items():
                print(f"Object Name: {name}, Count: {count}")          
        
        
        
        return processed_object_info_dict, aggregated_object_count_dict
    
    def _get_video_path(self, scene_idx: str, route_idx: str = None, view_count: int = 3) -> list:
        video_path_list = []
        scene_path = self._get_scene_path(scene_idx, route_idx)
        
        for view_id in range(view_count):
            view_name = f'cam{view_id}'
            video_path_list.append(os.path.join(scene_path, view_name, f"{view_name}.mp4"))
        return video_path_list
            
    def _get_raw_item(self, scene_idx: str, route_idx: str = None) -> tuple[list, list]:
        
        
        
        
        
        
        scene_path = self._get_scene_path(scene_idx, route_idx)
        
        view_dirs = sorted([x for x in Path(scene_path).iterdir() if x.is_dir() and x.name.startswith("cam")])
        
        video_framewise_object_ids_list = []
        video_path_list = []
        
        for view_dir in view_dirs:
            view_name = view_dir.name 
            video_path_list.append(os.path.join(view_dir, f"{view_name}.mp4"))
            
            
            npz_path = os.path.join(view_dir, "semantic_video_ids.npz")
            if os.path.exists(npz_path):
                video_framewise_object_ids_list.append(np.load(npz_path)['semantics'])
            
        
        
        return video_framewise_object_ids_list, video_path_list
    
    def get_common_object_ids(self, scene_idx: str, route_idx: str = None):
        video_framewise_object_ids_list, _ = self._get_raw_item(scene_idx, route_idx)
        if not video_framewise_object_ids_list:
            return set()
            
        
        video_unique_sets = [set(np.unique(v).tolist()) for v in video_framewise_object_ids_list]
        
        
        common_objects_ids = set.intersection(*video_unique_sets) if video_unique_sets else set()
        return common_objects_ids
    
    def get_object_name(self, scene_idx: str, route_idx: str = None, object_id: int = None):
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        return processed_object_info_dict.get(object_id, "Unknown Object")
    
    def get_common_object_names(self, scene_idx: str, route_idx: str = None, verbose: bool = False):
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx, verbose=verbose)
        common_objects_ids = self.get_common_object_ids(scene_idx, route_idx)
        
        common_objects_names = []
        for object_id in common_objects_ids:
            if object_id in processed_object_info_dict:
                name = processed_object_info_dict[object_id]
                common_objects_names.append(name)
                if verbose:
                    presence = self.get_timely_object_presence(scene_idx, route_idx, object_id)
                    print(f"ID {object_id}: {name} | Presence: {presence}")
        
        return common_objects_names 
    
    def get_timely_object_presence(self, scene_idx: str, route_idx: str = None, object_id: int = None):
        video_framewise_object_ids_list, _ = self._get_raw_item(scene_idx, route_idx)
        object_presence_dict = dict()
        
        for view_idx, framewise_object_ids_matrix in enumerate(video_framewise_object_ids_list):
            
            flat_view = framewise_object_ids_matrix.reshape((framewise_object_ids_matrix.shape[0], -1))
            
            frames_mask = np.any(flat_view == object_id, axis=1)
            
            
            object_presence_dict[f"cam{view_idx}"] = get_frame_ranges(np.where(frames_mask)[0].tolist())
        
        return object_presence_dict

    def get_objects_in_region(self, scene_idx: str, route_idx: str, region_id: str) -> list:
        """
        Retrieves a list of unique object names physically located inside a specific region.
        
        Args:
            scene_idx: The scene ID.
            route_idx: The route ID (e.g., 'route0').
            region_id: The ID of the region as a string (e.g., '5').
            
        Returns:
            list: A list of object names (e.g., ['chair', 'table', 'oven']).
        """
        
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        
        objects_in_region = set() 
        
        
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            
            
            if len(parts) >= 3:
                
                object_name = ' '.join(parts[1:-1])
                obj_region_id = parts[-1]
                
                
                if str(obj_region_id) == str(region_id):
                    
                    if object_name not in ["floor", "ceiling", "wall", "door", "window"]:
                        objects_in_region.add(object_name)
                        
        return list(objects_in_region)
    
    
    def guess_room_name(self, scene_idx: str, route_idx: str, region_id: str) -> str:
        """
        Identifies a region by its mathematically most unique object 
        (e.g., 'the room with the pipe' or 'the room with the oven').
        """
        objects_in_region = self.get_objects_in_region(scene_idx, route_idx, region_id)
        
        
        if not objects_in_region:
            return None
            
        
        _, global_counts = self.get_scene_object_info(scene_idx, route_idx)
        
        
        noise_words = ["trim", "molding", "baseboard", "unknown", "object", "frame"]
        
        valid_objects = []
        for obj in objects_in_region:
            if not any(noise in obj.lower() for noise in noise_words):
                valid_objects.append(obj)
                
        
        if not valid_objects:
            valid_objects = objects_in_region 
            
        
        valid_objects.sort(key=lambda x: global_counts.get(x, float('inf')))
        
        
        
        most_unique_object = valid_objects[0]
        
        if len(valid_objects) == 1:
            return None
        
        for idx in range(1, min(len(valid_objects), 3)):
            most_unique_object += f", {valid_objects[idx]}"
            
        return f"room with the {most_unique_object}"
    
    def debug_save_object_frames(self, scene_idx: str, route_idx: str, object_name: str, output_dir: str = "debug_frames"):
        """
        Finds an object by name in a specific scene/route, draws a bounding box 
        around it, and saves all video frames where that object is visible.
        """
        
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        
        
        target_ids = []
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            if len(parts) >= 3:
                name = ' '.join(parts[1:-1])
                if name.lower() == object_name.lower():
                    target_ids.append(obj_id)
                    
        if not target_ids:
            print(f"Debug: Object '{object_name}' not found in Scene {scene_idx}, Route {route_idx}.")
            return
            
        print(f"Debug: Found '{object_name}' with ID(s): {target_ids}")
        
        
        video_framewise_object_ids_list, video_path_list = self._get_raw_item(scene_idx, route_idx)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        
        
        for view_idx, (semantic_video, video_path) in enumerate(zip(video_framewise_object_ids_list, video_path_list)):
            
            
            flat_view = semantic_video.reshape((semantic_video.shape[0], -1))
            frames_mask = np.isin(flat_view, target_ids).any(axis=1)
            frame_indices = np.where(frames_mask)[0]
            
            if len(frame_indices) == 0:
                continue
                
            print(f"Debug: Extracting and annotating {len(frame_indices)} frames from cam{view_idx}...")
            
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Debug Error: Failed to open video at {video_path}")
                continue
                
            current_frame_idx = 0
            frames_saved = 0
            max_frame_needed = frame_indices[-1] 
            
            while True:
                ret, frame = cap.read()
                if not ret or current_frame_idx > max_frame_needed:
                    break
                    
                if current_frame_idx in frame_indices:
                    
                    current_mask = semantic_video[current_frame_idx]
                    
                    
                    for t_id in target_ids:
                        
                        y_coords, x_coords = np.where(current_mask == t_id)
                        
                        if len(y_coords) > 0:
                            
                            total_pixels = current_mask.size
                            object_pixels = len(y_coords)
                            occupancy_pct = (object_pixels / total_pixels) * 100
                            
                            
                            
                            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                            
                            
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            
                            
                            label = f"{object_name} (ID: {t_id}) | {occupancy_pct:.1f}%"
                            cv2.putText(frame, label, (x_min, max(y_min - 5, 0)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    safe_obj_name = object_name.replace(' ', '_')
                    filename = f"{scene_idx}_{route_idx}_cam{view_idx}_frame{current_frame_idx:04d}_{safe_obj_name}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(save_path, frame)
                    frames_saved += 1
                    
                current_frame_idx += 1
                
            cap.release()
            print(f"Debug: Saved {frames_saved} annotated frames for cam{view_idx}.")
            
        print(f"Debug: Finished saving all frames for '{object_name}' to ./{output_dir}/")
    
    

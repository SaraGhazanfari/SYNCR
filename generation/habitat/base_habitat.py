from utils import DATA_ROOT
import os
import numpy as np
import json
from pathlib import Path
import random
from utils import write_jsonl_file, read_json_file, read_jsonl_file
import cv2
import shutil
import argparse


objects_to_filter = {
    # Tiny / Small / Thin
    'glass', 'cup', 'plate', 'bowl', 'decorative bowl', 'bowl of sweets', 'bowl of fruit', 
    'bottle', 'vase', 'flower vase', 'flowerpot', 'flower', 'candle', 'toy', 'ornament', 
    'hook', 'remote control', 'computer mouse', 'keyboard', 'device', 'tablet', 'sensor', 
    'fire alarm', 'paper', 'note', 'book', 'magazine', 'newspaper', 'binder', 'door knob', 
    'door hinge', 'self-closing mechanism', 'faucet', 'tap', 'clothes hanger', 'knife holder', 
    'ceiling light fixture connection', 'shoe', 'hat', 'bag', 'backpack',
    
    # Clutter / Amorphous / Uncountable
    'clothes', 'blanket', 'throw blanket', 'towel', 'pillow', 'cushion', 'clutter', 
    'shelf with clutter', 'kitchen countertop item', 'decoration', 'kitchen decoration', 
    'wall hanging decoration', 'unknown', 'box', 'tray', 'pot', 'appliance', 'kitchen appliance',
    'kitchen counter', 
    
    # Continuous / Structural / Ambiguous
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


def get_sampled_frame_index(frame_id, total_frames, nframes=32):
    """
    Maps a raw video frame index to the position it lands on among the
    nframes actually sampled by qwen_vl_utils at train/inference time:
    torch.linspace(0, total_frames - 1, nframes).round() (verified directly
    against the installed qwen_vl_utils source, not assumed -- see the
    equivalent CLEVRER helper, generation/clevrer/utils.py's
    get_sampled_frame_index, for the verification). Unlike CLEVRER's fixed
    128-frame videos, Habitat videos have a per-video native frame count
    (walking distance varies per route), so total_frames must be the actual
    frame count of the specific video being referenced, not a shared
    constant.
    """
    if nframes <= 1 or total_frames <= 1:
        return 0
    return round(frame_id * (nframes - 1) / (total_frames - 1))


class Habitat_Dataset:
    def __init__(self, root_path, section, split):
        self.data_path = os.path.join(root_path, section, split)
        self.data_type = section
        self.root_path = root_path
        self.split = split 
        # self.raw_data_dict = dict()
        # self.obj_info_dict = dict()
        
    def _get_scene_path(self, scene_idx: str, route_idx: str = None):
        scene_path = os.path.join(self.data_path, scene_idx)
        if route_idx:
            scene_path = os.path.join(scene_path, route_idx)
        return scene_path

    def _get_semantic_path(self, scene_idx: str, route_idx: str = None):
        return os.path.join(self._get_scene_path(scene_idx, route_idx),
                            'scene_semantic_objects.json')

    def get_scene_object_info(self, scene_idx: str, route_idx: str = None, verbose: bool = False) -> tuple[dict, dict]:
        # Cache key uses route_idx to differentiate data within the same scene
        # cache_key = (scene_idx, route_idx)
        # if cache_key in self.obj_info_dict:
        #     return self.obj_info_dict[cache_key]
        
        # if len(self.obj_info_dict) >= 500:
        #     self.obj_info_dict = dict()
                  
        semantic_path = self._get_semantic_path(scene_idx, route_idx)
        raw_object_info_dict = read_json_file(semantic_path)
        processed_object_info_dict = dict()
        aggregated_object_count_dict = dict()
        
        for key, value in raw_object_info_dict.items():
            if key == "0" or ' ' not in value: 
                continue
         
            # Standardizing name extraction
            value_parts = value.split(' ')
            object_name = ' '.join(value_parts[1:-1])
            
            if object_name in ["floor", "ceiling", "wall", "objects"]:
                continue
                
            aggregated_object_count_dict[object_name] = aggregated_object_count_dict.get(object_name, 0) + 1
            
            # Update the count in the string (e.g., "1 chair" becomes "2 chair")
            value_parts[0] = str(aggregated_object_count_dict[object_name])
            processed_object_info_dict[int(key)] = ' '.join(value_parts)
        
        if verbose:
            print(f"Scene {scene_idx} Route {route_idx} objects:")
            for name, count in aggregated_object_count_dict.items():
                print(f"Object Name: {name}, Count: {count}")          
        
        # self.obj_info_dict[cache_key] = (processed_object_info_dict, aggregated_object_count_dict)
        # return self.obj_info_dict[cache_key]
        return processed_object_info_dict, aggregated_object_count_dict
    
    def _get_video_path(self, scene_idx: str, route_idx: str = None, view_count: int = 3) -> list:
        video_path_list = []
        scene_path = self._get_scene_path(scene_idx, route_idx)
        
        for view_id in range(view_count):
            view_name = f'cam{view_id}'
            video_path_list.append(os.path.join(scene_path, view_name, f"{view_name}.mp4"))
        return video_path_list
            
    def _get_raw_item(self, scene_idx: str, route_idx: str = None) -> tuple[list, list]:
        # cache_key = (scene_idx, route_idx)
        # if cache_key in self.raw_data_dict:
        #     return self.raw_data_dict[cache_key]
        
        # if len(self.raw_data_dict) >= 500:
        #     self.raw_data_dict = dict()
        scene_path = self._get_scene_path(scene_idx, route_idx)
        # Only count directories that match the 'camX' pattern
        view_dirs = sorted([x for x in Path(scene_path).iterdir() if x.is_dir() and x.name.startswith("cam")])
        
        video_framewise_object_ids_list = []
        video_path_list = []
        
        for view_dir in view_dirs:
            view_name = view_dir.name # e.g., "cam0"
            video_path_list.append(os.path.join(view_dir, f"{view_name}.mp4"))
            
            # Load the semantic labels for this view
            npz_path = os.path.join(view_dir, "semantic_video_ids.npz")
            if os.path.exists(npz_path):
                video_framewise_object_ids_list.append(np.load(npz_path)['semantics'])
            
        # self.raw_data_dict[cache_key] = (video_framewise_object_ids_list, video_path_list)    
        # return self.raw_data_dict[cache_key]
        return video_framewise_object_ids_list, video_path_list
    
    def get_common_object_ids(self, scene_idx: str, route_idx: str = None):
        video_framewise_object_ids_list, _ = self._get_raw_item(scene_idx, route_idx)
        if not video_framewise_object_ids_list:
            return set()
            
        # Get unique IDs for each camera view
        video_unique_sets = [set(np.unique(v).tolist()) for v in video_framewise_object_ids_list]
        
        # Intersection across all available views
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
    
    def get_video_frame_counts(self, scene_idx: str, route_idx: str = None):
        """{cam_key: native frame count}, for mapping a raw frame index into
        the sampled-frame space (get_sampled_frame_index) -- Habitat videos
        vary in length per route, unlike CLEVRER's fixed 128 frames, so this
        can't be a shared constant."""
        video_framewise_object_ids_list, _ = self._get_raw_item(scene_idx, route_idx)
        return {
            f"cam{view_idx}": matrix.shape[0]
            for view_idx, matrix in enumerate(video_framewise_object_ids_list)
        }

    def get_timely_object_presence(self, scene_idx: str, route_idx: str = None, object_id: int = None):
        video_framewise_object_ids_list, _ = self._get_raw_item(scene_idx, route_idx)
        object_presence_dict = dict()
        
        for view_idx, framewise_object_ids_matrix in enumerate(video_framewise_object_ids_list):
            # Reshape to (Frames, H*W) to check if the ID exists anywhere in the frame
            flat_view = framewise_object_ids_matrix.reshape((framewise_object_ids_matrix.shape[0], -1))
            # Boolean mask: True if object_id is in that frame
            frames_mask = np.any(flat_view == object_id, axis=1)
            
            # Convert boolean mask to list of (start, end) ranges
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
        # 1. Fetch the processed dictionary for this scene/route
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        
        objects_in_region = set() # Use a set to automatically prevent duplicates
        
        # 2. Iterate through all objects to find ones matching our target region
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            
            # Ensure the string has enough parts to contain a count, name, and region
            if len(parts) >= 3:
                # Extract the name and the region ID
                object_name = ' '.join(parts[1:-1])
                obj_region_id = parts[-1]
                
                # Check if this object belongs to our target region
                if str(obj_region_id) == str(region_id):
                    # Filter out structural noise so we only get useful room-defining furniture
                    if object_name not in ["floor", "ceiling", "wall", "door", "window"]:
                        objects_in_region.add(object_name)
                        
        return list(objects_in_region)

    def get_region_object_ids(self, scene_idx: str, route_idx: str, region_id: str) -> set:
        """
        Same lookup as get_objects_in_region, but keeps the object ids instead of
        collapsing to names -- needed to check a region's objects against
        semantic_video_ids.npz (which is id-indexed) for chain-of-frames grounding
        in Route Planning: Route Planning never recorded which rendered frame the
        camera was in each region (the walk is pure point-to-point pathfinding, no
        region tracking during rendering), and re-rendering to add that would be
        expensive. Since regions are already identified by their characteristic
        objects (guess_room_name), using ALL of a region's object ids (not just the
        single most-distinctive one) as evidence for "this frame is likely region
        X" reuses data that's already saved, at the cost of being a proxy rather
        than ground truth -- reasonable for enrichment evidence in a reasoning
        trace, not for the graded answer itself (which is untouched by this).
        """
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        obj_ids = set()
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            if len(parts) >= 3:
                object_name = ' '.join(parts[1:-1])
                obj_region_id = parts[-1]
                if str(obj_region_id) == str(region_id) and object_name not in ["floor", "ceiling", "wall", "door", "window"]:
                    obj_ids.add(obj_id)
        return obj_ids

    def guess_room_name(self, scene_idx: str, route_idx: str, region_id: str) -> str:
        """
        Identifies a region by its mathematically most unique object 
        (e.g., 'the room with the pipe' or 'the room with the oven').
        """
        objects_in_region = self.get_objects_in_region(scene_idx, route_idx, region_id)
        
        # If the room has no defining objects at all, fallback to the ID
        if not objects_in_region:
            return None
            
        # Fetch the global counts of all objects across the entire scene
        _, global_counts = self.get_scene_object_info(scene_idx, route_idx)
        
        # Optional: Filter out structural noise that might technically be rare but makes for bad labels
        noise_words = ["trim", "molding", "baseboard", "unknown", "object", "frame"]
        
        valid_objects = []
        for obj in objects_in_region:
            if not any(noise in obj.lower() for noise in noise_words):
                valid_objects.append(obj)
                
        # If our filter accidentally removed everything, revert to the original list
        if not valid_objects:
            valid_objects = objects_in_region 
            
        # Sort the room's objects by how rare they are in the entire apartment (Lowest count first)
        valid_objects.sort(key=lambda x: global_counts.get(x, float('inf')))
        
        # The first item is now the rarest, most unique object in that specific room!
        
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
        # 1. Fetch the processed dictionary for this scene/route
        processed_object_info_dict, _ = self.get_scene_object_info(scene_idx, route_idx)
        
        # 2. Find all object IDs that match the requested object_name
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
        
        # 3. Get the semantic maps and video paths
        video_framewise_object_ids_list, video_path_list = self._get_raw_item(scene_idx, route_idx)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 4. Iterate through each camera view
        for view_idx, (semantic_video, video_path) in enumerate(zip(video_framewise_object_ids_list, video_path_list)):
            
            # Flatten spatial dimensions to find which frames contain the target IDs
            flat_view = semantic_video.reshape((semantic_video.shape[0], -1))
            frames_mask = np.isin(flat_view, target_ids).any(axis=1)
            frame_indices = np.where(frames_mask)[0]
            
            if len(frame_indices) == 0:
                continue
                
            print(f"Debug: Extracting and annotating {len(frame_indices)} frames from cam{view_idx}...")
            
            # 5. Extract, annotate, and save the frames
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
                    # Get the 2D semantic mask just for this specific frame
                    current_mask = semantic_video[current_frame_idx]
                    
                    # Draw bounding boxes for each instance ID of the object
                    for t_id in target_ids:
                        # Find the spatial (y, x) coordinates where this object exists in the mask
                        y_coords, x_coords = np.where(current_mask == t_id)
                        
                        if len(y_coords) > 0:
                            # --- NEW: Calculate occupancy percentage ---
                            total_pixels = current_mask.size
                            object_pixels = len(y_coords)
                            occupancy_pct = (object_pixels / total_pixels) * 100
                            # -------------------------------------------
                            
                            # Get the bounding box min/max coordinates
                            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                            
                            # Draw a bright green bounding box (BGR format)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            
                            # Add a text label right above the box with the percentage
                            label = f"{object_name} (ID: {t_id}) | {occupancy_pct:.1f}%"
                            cv2.putText(frame, label, (x_min, max(y_min - 5, 0)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Save the annotated frame
                    safe_obj_name = object_name.replace(' ', '_')
                    filename = f"{scene_idx}_{route_idx}_cam{view_idx}_frame{current_frame_idx:04d}_{safe_obj_name}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(save_path, frame)
                    frames_saved += 1
                    
                current_frame_idx += 1
                
            cap.release()
            print(f"Debug: Saved {frames_saved} annotated frames for cam{view_idx}.")
            
        print(f"Debug: Finished saving all frames for '{object_name}' to ./{output_dir}/")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models.")
    
    # Add command-line arguments
    parser.add_argument("--idx", type=int, help="The model version/path to evaluate.")
    
    args = parser.parse_args()
    
    path = f"{DATA_ROOT}/habitat_data"
    data = read_jsonl_file("json_files/eval/Holistic_Synthesis/Habitat_Object_Counting.jsonl")
    scene_idx = data[args.idx
                     ]['video_list'][0]['video_path'].replace(
                         f"{DATA_ROOT}/habitat_data/route_plan/val/", '').split('/')[0]
    object_name = data[args.idx]['question'].split("'")[1]
    route_idx = data[args.idx]['video_list'][0]['video_path'].replace(
                         f"{DATA_ROOT}/habitat_data/route_plan/val/", '').split('/')[1]
    print(f"Debug: Extracted scene_idx='{scene_idx}' and object_name='{object_name}' from the question, answer={data[args.idx]['answer']}.")
    habitat_obj = Habitat_Dataset(root_path=path, section='route_plan', split='val')
    out = habitat_obj.debug_save_object_frames(scene_idx=scene_idx, 
                                               route_idx=route_idx, object_name=object_name, 
                                               output_dir="debug_frames")
    
    
    
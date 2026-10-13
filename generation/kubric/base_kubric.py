from utils import DATA_ROOT
import os
import random
import numpy as np
import cv2

from utils import read_json_file, write_jsonl_file, save_video_frame
from generation.option_balance import reoption_spatial_v3, sync_distractors


class Kubric_Dataset:
    def __init__(self, path, view_count=3):
        self.path = path
        self.view_count = view_count   
        self.instance_data = dict()
        
    def _get_raw_item(self, scene_idx: str):
        '''
        - 'camera': This key holds all the physical and optical 
                    properties of the camera used to render that 
                    specific video.
                    
        - 'instances':This is the absolute goldmine of the file. 
                    It contains a list (or dictionary) of every single 
                    3D object (instance) generated in the scene.
                    
        - 'flags': This key contains the configuration settings and 
                 command-line arguments that were passed to the Kubric 
                 worker to generate this specific scene.
                 
        - 'metadata': This key holds general administrative information 
                      about the simulation run.
        '''
        if scene_idx in self.instance_data:
            return self.instance_data[scene_idx]
        
        data_dict = dict()
        scene_path = os.path.join(self.path, scene_idx)
        
        for i in range(self.view_count):
            metadata_path = os.path.join(scene_path, f"cam{i}", "metadata.json")
            data_dict[f"cam{i}"] = read_json_file(metadata_path)['instances']
            
        self.instance_data[scene_idx] = data_dict
        return data_dict
    
    def _count_object_occurences(self, scene_idx, all_objects_in_scene):
        name_counts = {}
        for obj in all_objects_in_scene:
            base_name = self.get_object_name(scene_idx, obj)
            name_counts[base_name] = name_counts.get(base_name, 0) + 1
        return name_counts
        
        
    def get_visibility_for_all_objects(self, scene_idx, view_count, cams=None):
        # 1. Grab cam0 to determine the baseline dimensions
        if scene_idx not in self.instance_data:
            self._get_raw_item(scene_idx)

        cam0_instances = self.instance_data[scene_idx]['cam0']

        # Because the rest of your code relies on cam0's exact list index,
        # we MUST treat cam0 as the master template for object count.
        object_count = len(cam0_instances)
        frame_count = len(cam0_instances[0]['visibility'])

        # 2. Build a mapping of unique IDs to cam0's list indices
        cam0_id_map = {}
        for idx, obj in enumerate(cam0_instances):
            unique_id = obj['asset_id']

            if unique_id is not None:
                cam0_id_map[unique_id] = idx

        # cams lets callers pick which physical cameras fill each view
        # position (e.g. [1, 2] instead of the default [0, 1, ..]) --
        # used to probe a different camera pair than the shipped benchmark.
        if cams is None:
            cams = list(range(view_count))

        # 3. Initialize visibility array: (views, objects, frames)
        visibility = np.zeros((len(cams), object_count, frame_count))

        # 4. Populate visibility using the mapping
        for view_pos, cam_id in enumerate(cams):
            cam_name = f"cam{cam_id}"

            for object_dict in self.instance_data[scene_idx][cam_name]:
                unique_id = object_dict['asset_id']

                # Find where this object belongs based on cam0's indexing
                if unique_id in cam0_id_map:
                    mapped_idx = cam0_id_map[unique_id]
                    visibility[view_pos, mapped_idx, :] = object_dict['visibility']
                else:
                    # Edge case handling: If an object appears in cam1 but is totally
                    # absent from cam0's JSON, we safely ignore it. If we included it,
                    # your downstream code would crash trying to look it up in cam0.
                    pass

        return visibility
    
    def get_time_sync_qa(self, scene_idx: str, fps=12, original_duration=5, crop_duration=3):
        """
        Randomly defines 3-second crop windows from 5-second original videos.
        Allows for negative offsets by randomly assigning the reference video.
        Outputs a dataset sample dict.
        """
        scene_path = os.path.join(self.path, scene_idx)
        
        # 1. Calculate frame boundaries
        total_frames = int(original_duration * fps)
        crop_frames = int(crop_duration * fps)      
        
        max_start_frame = total_frames - crop_frames
        
        # 2. Sample 3 unique start frames (NO SORTING)
        # Because we don't sort, t1 might be greater than t2 or t3, 
        # naturally resulting in negative time offsets.
        start_frames = random.sample(range(0, max_start_frame + 1), 3)
        t1, t2, t3 = start_frames
        
        # 3. Assign physical cameras to these time slots
        cameras = ['cam0', 'cam1', 'cam2']
        random.shuffle(cameras)
        cam_vid1, cam_vid2, cam_vid3 = cameras
        
        # 4. Calculate exact offsets in seconds relative to Video 1 (t1)
        # If t2 < t1, this math naturally produces a negative number.
        offset1_sec = (t2 - t1) / float(fps)
        offset2_sec = (t3 - t1) / float(fps)
        
        # 5. Build the video list instruction set
        video_list = [
            {"video_path": os.path.join(scene_path, f"{cam_vid1}.mp4"), "start_frame": t1, "end_frame": t1 + crop_frames,
             "start_sec": round(t1 / float(fps), 1), "end_sec": round((t1 + crop_frames) / float(fps), 1)},
            {"video_path": os.path.join(scene_path, f"{cam_vid2}.mp4"), "start_frame": t2, "end_frame": t2 + crop_frames,
             "start_sec": round(t2 / float(fps), 1), "end_sec": round((t2 + crop_frames) / float(fps), 1)},
            {"video_path": os.path.join(scene_path, f"{cam_vid3}.mp4"), "start_frame": t3, "end_frame": t3 + crop_frames,
             "start_sec": round(t3 / float(fps), 1), "end_sec": round((t3 + crop_frames) / float(fps), 1)}
        ]
        
        # 6. Formulate the correct answer string using {:+.1f}
        # This automatically formats positive numbers with '+' and negative with '-'
        correct_text = f"Video 2 starts at {offset1_sec:+.1f} seconds; Video 3 starts at {offset2_sec:+.1f} seconds."
        
        # 7. Distractors: offsets from independent re-runs of the crop sampling
        # above. The old swapped / sign-flipped / fuzzed copies of the answer made
        # it the only option with both twins (see generation/option_balance.py).
        options = [correct_text] + sync_distractors(correct_text, random, fps=fps,
                                                    original_duration=original_duration,
                                                    crop_duration=crop_duration)
        
        # 8. Shuffle options and assign multiple choice letters
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        options_list = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
        
        # Find the letter corresponding to the correct answer
        correct_label = ""
        for option in options_list:
            if option.split(') ')[1] == correct_text:
                correct_label = option
                break
                
        # 9. Construct the static question
        question = (
            "You are provided with three videos (Video 1, Video 2, and Video 3) recorded at {fps} FPS. "
            f"Each video is exactly {crop_duration}.0 seconds long and shows the same physical event "
            "captured simultaneously from three different camera angles.\n"
            "Identify the exact temporal offset of the latter two videos. At what timestamp in Video 1's timeline "
            "does the very first frame of Video 2 and Video 3 occur? (Note: A negative timestamp means the video started before Video 1)."
        )
        
        # 10. Assemble and return the final sample dictionary
        sample = {
            'video_list': video_list,
            'question': question,
            'options': options_list,
            'answer': correct_label
        }

        # Structured ground truth for the GRPO process reward (training code, not released):
        # the two temporal offsets in seconds, rounded to the same 1-decimal
        # precision the answer options themselves use. Numeric (not a
        # phrase), so process_reward auto-routes it through tolerance-based
        # number matching instead of substring matching -- an exact-string
        # check would almost never fire, since a model reasoning in its own
        # words will rarely emit the precise substring "+0.4" rather than
        # e.g. "approximately 0.4 seconds" or "0.38s".
        sample['process_gt'] = {
            'video_2_offset_sec': round(offset1_sec, 1),
            'video_3_offset_sec': round(offset2_sec, 1),
        }

        return sample

    def get_hit_frames(self, scene_idx, all_unique_objects):
        hit_frames = {}
        for obj_idx in all_unique_objects:
            positions = np.array(self.instance_data[scene_idx]['cam0'][obj_idx]['positions'])
            z_coords = positions[:, 2] 
            z_velocity = np.diff(z_coords)
            impacts = np.where((z_velocity[:-1] < 0) & (z_velocity[1:] >= 0))[0]
            
            if len(impacts) > 0:
                hit_frames[obj_idx] = impacts[0] + 1 
            else:
                hit_frames[obj_idx] = None
                
        return hit_frames

    def get_objects_visible_in_all_views(self, scene_idx, view_count):
        visibility = self.get_visibility_for_all_objects(scene_idx, view_count=view_count)

        visible_in_all_views = np.all(visibility > 0, axis=0) 

        # 2. To get a list of actual object indices per frame:
        objects_per_frame = [np.where(visible_in_all_views[:, f])[0].tolist() 
                            for f in range(visible_in_all_views.shape[1])]

        all_unique_objects = set(np.where(visible_in_all_views.any(axis=1))[0])
        return all_unique_objects
    
    def rank_objects_by_distance(self, scene_idx, target_obj, event_obj, exit_frame, visibility, view_count):
        """Objects other than target/event visible (> 25 px in any view) at exit_frame,
        as [(object_idx, distance to target_obj)], closest first."""
        instances = self.instance_data[scene_idx]['cam0']
        target_pos = np.array(instances[target_obj]['positions'][exit_frame])
        distances = {}
        for other_obj in range(visibility.shape[1]):
            if other_obj == target_obj or other_obj == event_obj:
                continue
            # Require at least 25 visible pixels so the VLM can actually see it
            if not any(visibility[v, other_obj, exit_frame] > 25 for v in range(view_count)):
                continue
            other_pos = np.array(instances[other_obj]['positions'][exit_frame])
            distances[other_obj] = np.linalg.norm(target_pos - other_pos)
        return sorted(distances.items(), key=lambda item: item[1])

    def get_object_name(self, scene_idx: str, object_idx: int) -> str:
        """
        Retrieves a detailed, unique name for an object (e.g., 'small red metal cube (ID: 2)').
        """
        if scene_idx not in self.instance_data:
            self._get_raw_item(scene_idx)
        
        object_dict = self.instance_data[scene_idx]['cam0'][object_idx]
        # 1. Base shape
        shape = str(object_dict.get('asset_id', 'object')).split('_')[-1] 
        
        # 2. Material
        material = str(object_dict.get('material', '')).split('_')[-1]
        
        # 3. Color (ignore if it's a raw RGB array)
        color = object_dict.get('color_label', '')
        if isinstance(color, (list, np.ndarray)):
            color = "" 
            
        # 4. --- NEW: Size Logic ---
        size = object_dict.get('size', object_dict.get('scale', ''))
        if isinstance(size, (int, float)):
            # If your dataset uses numeric scales, you can threshold them into words.
            # (Adjust the 1.0 threshold based on your specific dataset's baseline scale)
            size = "small" if size < 1.0 else "large"
        elif isinstance(size, (list, np.ndarray)):
            # If scale is a 3D vector [x, y, z], use the first dimension or an average
            size_val = size[0] 
            size = "small" if size_val < 1.0 else "large"
            
        # 5. Combine descriptors in natural English order: Size -> Color -> Material -> Shape
        descriptors = [str(d) for d in [size, color, material, shape] if d]
        detailed_name = " ".join(descriptors).strip()
        
        # 6. Guarantee uniqueness
        return detailed_name.title()


    def get_cross_video_exit_qa(self, scene_idx: str, view_count: int = 2, cams=None,
                                all_targets=False, sample_by_exit_time=True, allow_duplicate_names=False):
        """Spatial Measurement items for one camera ordering, answer option only.

        Distractors are filled in over the whole list by finalize_spatial_options.
        all_targets=False keeps the first valid target per event object's exit (as the
        benchmark was built), True keeps every valid target. sample_by_exit_time=True
        applies the benchmark's early/mid/late exit-frame sampling, False returns all items.
        allow_duplicate_names=False skips scenes where any two objects share a name (as the
        benchmark was built); True keeps them but uses only uniquely named event, target,
        answer and distractor objects.
        """
        # 1. Ensure the data is loaded
        if scene_idx not in self.instance_data:
            self._get_raw_item(scene_idx)

        # cams picks which physical cameras form the video pair (default:
        # cam0+cam1, matching the shipped benchmark). Passing e.g. [1, 2]
        # probes an entirely different camera pair -- useful for held-out
        # training scenes where every scene productive under cam0+cam1 has
        # already been consumed by the benchmark.
        if cams is None:
            cams = list(range(view_count))
        view_count = len(cams)

        scene_path = os.path.join(self.path, scene_idx)
        video_list = [{'video_path': os.path.join(scene_path, f"cam{c}.mp4")} for c in cams]

        visibility = self.get_visibility_for_all_objects(scene_idx, view_count, cams=cams)
        num_views, num_objects, frame_count = visibility.shape
        
        all_objects_in_scene = list(range(num_objects))
        name_counts = self._count_object_occurences(scene_idx, all_objects_in_scene) 
        
        if not allow_duplicate_names and any(count > 1 for count in name_counts.values()):
            # Optional: print statement for debugging so you know it was skipped
            print(f"Skipping {scene_idx}: Found duplicate objects.")
            return []
        
        qa_dataset_samples = []
        
        # 2. Iterate to find an Event Object (visible in ONLY ONE view)
        for event_obj in all_objects_in_scene:
            event_name = self.get_object_name(scene_idx, event_obj)
            if name_counts[event_name] > 1:
                continue
                
            # Check which views this object actually appears in
            views_visible_in = [v for v in range(view_count) if visibility[v, event_obj, :].sum() > 0]
            
            # # CONDITION 1: Event object must ONLY be in one video
            if len(views_visible_in) == 0:
                continue
             
            event_cam_id = views_visible_in[0]
            vis_array = visibility[event_cam_id, event_obj, :]
            
            # Find the exit frame for this event object
            exits = np.where((vis_array[:-1] > 0) & (vis_array[1:] == 0))[0]
            if len(exits) == 0:
                continue
                
            exit_frame = exits[0] + 1
            if exit_frame >= frame_count or exit_frame < 12:
                continue # Don't use the absolute last frame
                
            video_number = event_cam_id + 1
            
            # 3. Find a Target Object (visible in ALL views at `exit_frame`)
            for target_obj in all_objects_in_scene:
                if target_obj == event_obj:
                    continue
                    
                target_name = self.get_object_name(scene_idx, target_obj)
                if name_counts[target_name] > 1:
                    continue
                    
                # CONDITION 2: Target object must be clearly visible in ALL videos at this exact frame
                visible_in_all = all(visibility[v, target_obj, exit_frame] > 150 for v in range(view_count))
                if not visible_in_all:
                    continue
                
                # 4-6. Visible objects ranked by distance to the TARGET; the closest is the answer
                sorted_by_distance = self.rank_objects_by_distance(scene_idx, target_obj, event_obj,
                                                                   exit_frame, visibility, view_count)
                if not sorted_by_distance:
                    continue
                closest_obj = sorted_by_distance[0][0]
                if name_counts[self.get_object_name(scene_idx, closest_obj)] > 1:
                    continue  # the answer must name one object
                correct_answer = f"{self.get_object_name(scene_idx, closest_obj)}"

                # 7. Question. Options hold only the answer here: distractors are drawn over the
                # whole list by finalize_spatial_options (reoption_spatial_v3), matched to the
                # answers' attributes (v2: answers more often "Small", never the target's shape).
                question = (
                    f"Observe the videos closely. At the exact moment the {event_name} completely exits the frame "
                    f"in Video {video_number}, which object is physically closest to the {target_name} in the 3D space?"
                )
                options_list = [f"A) {correct_answer}"]
                correct_label = options_list[0]
                    
                qa_dataset_samples.append({
                    'scene_idx': scene_idx,
                    'event_object': self.get_object_name(scene_idx, event_obj),
                    'target_object': self.get_object_name(scene_idx, target_obj),
                    'exit_frame': str(exit_frame),
                    'video_condition': f"Video {video_number}",
                    'video_list': video_list,
                    'question': question,
                    'options': options_list,
                    'answer': correct_label,
                    '_spatial_candidates': {  # popped by finalize_spatial_options
                        'ranked': [self.get_object_name(scene_idx, obj) for obj, _ in sorted_by_distance
                                   if name_counts[self.get_object_name(scene_idx, obj)] == 1],
                        'scene_names': [self.get_object_name(scene_idx, obj) for obj in all_objects_in_scene
                                        if name_counts[self.get_object_name(scene_idx, obj)] == 1],
                    },
                    # Process reward ground truth: the event object and target
                    # object names -- the two intermediate identities the
                    # question requires locating before distance can even be
                    # compared -- not exit_frame (no established fps for this
                    # function's videos to convert it to a seconds value
                    # comparable across scenes, unlike get_time_sync_qa's
                    # explicit fps param) and not closest_obj (that IS the
                    # answer -- rewarding it would let a model get credit for
                    # restating its chosen option, the exact bug the GRPO reward's
                    # process_reward docstring already warns about).
                    'process_gt': {
                        'event_object': self.get_object_name(scene_idx, event_obj),
                        'target_object': target_name,
                    },
                })
                
                # One question per event object's exit, unless every valid target is wanted
                if not all_targets:
                    break
                
        if not sample_by_exit_time:
            return qa_dataset_samples

        # --- Distribute and Sample ---
        bucket_early = []  
        bucket_mid = []    
        bucket_late = []   
        
        for sample in qa_dataset_samples:
            if int(sample['exit_frame']) <= 10:
                bucket_early.append(sample)
            elif 10 < int(sample['exit_frame']) <= 20:
                bucket_mid.append(sample)
            else:
                bucket_late.append(sample)
                
        random.shuffle(bucket_early)
        random.shuffle(bucket_mid)
        
        final_samples = []
        if bucket_early: final_samples.extend(bucket_early[:1]) 
        if bucket_mid: final_samples.extend(bucket_mid[:2])   
        final_samples.extend(bucket_late) 
        
        return final_samples
    
def finalize_spatial_options(qa_list, rng):
    """Fill in the distractors of items from get_cross_video_exit_qa, over the whole list.

    Uses reoption_spatial_v3 (distractor attributes matched to the answers') and
    pops each item's '_spatial_candidates'. Returns the builder's stats.
    """
    candidates = [row.pop('_spatial_candidates') for row in qa_list]
    items = []
    for row, cand in zip(qa_list, candidates):
        answer = row['answer'][3:]
        items.append({'answer': answer, 'target': row['target_object'], 'event': row['event_object'],
                      'ranked': [name for name in cand['ranked'] if name != answer],
                      'scene_names': cand['scene_names']})
    built, stats = reoption_spatial_v3(items, rng)
    for row, (options, answer) in zip(qa_list, built):
        row['options'], row['answer'] = options, answer
    return stats


def get_multi_angle_sync_qa(total_num=1000, save=False):
    data_obj = Kubric_Dataset(path=f"{DATA_ROOT}/kubric_data")
    data_list = list()

    for i in range(total_num):
        data_list.append(data_obj.get_time_sync_qa(scene_idx=f"scene_{i}"))


    eval_path = 'json_files/eval/Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl'
    if save:
        write_jsonl_file(filepath=eval_path, data_list=data_list)
    

def get_cross_video_spatial_measurement_qa(total_num=1000, save=False):  
    data_obj = Kubric_Dataset(path=f"{DATA_ROOT}/kubric_data")
    qa_list = list()
    
    for idx, scene_idx in enumerate(os.listdir(data_obj.path)):
        if not scene_idx.startswith("scene_"):
            continue
        scene_samples = data_obj.get_cross_video_exit_qa(scene_idx=scene_idx, view_count=2)
        qa_list.extend(scene_samples)
        print(f"{idx + 1}/{len(os.listdir(data_obj.path))} -Total samples collected: {len(qa_list)}")
        
        if len(qa_list) >= total_num:
            break
            
    if qa_list:
        print("Sample 0:", qa_list[0])  
        
    else:
        print("No samples were generated!")
     
    eval_path = 'json_files/eval/Spatial_Tracking/Kubric_Cross_Video_Spatial_Measurement.jsonl'
    
    # Optional: ensure directory exists before saving JSONL
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    
    qa_list = qa_list[:total_num]
    if qa_list:
        print("Spatial options:", finalize_spatial_options(qa_list, random))
    if save:
        write_jsonl_file(filepath=eval_path, data_list=qa_list)
    
# get_multi_angle_sync_qa(total_num=1000, save=False)
# get_cross_video_spatial_measurement_qa(total_num=1000, save=True)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kubric question pools (original benchmark files)")
    parser.add_argument("--task", choices=["spatial", "sync"], default="spatial")
    parser.add_argument("--total-num", type=int, default=1000)
    args = parser.parse_args()
    random.seed(0)
    if args.task == "spatial":
        get_cross_video_spatial_measurement_qa(total_num=args.total_num, save=True)
    else:
        get_multi_angle_sync_qa(total_num=args.total_num, save=True) 
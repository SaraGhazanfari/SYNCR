import os
import random
import argparse
import numpy as np

from kubric.base_kubric import KubricDataset
from utils import write_jsonl_file

class SpatialMeasurementQAGenerator:
    def __init__(self, data_manager: KubricDataset):
        self.data_manager = data_manager

    def generate(self, scene_idx: str, view_count: int = 2):
        self.data_manager.get_raw_item(scene_idx) # Ensure loaded
        scene_path = os.path.join(self.data_manager.path, scene_idx)
        video_list = [{'video_path': os.path.join(scene_path, f"cam{idx}.mp4")} for idx in range(view_count)]    
        
        visibility = self.data_manager.get_visibility_for_all_objects(scene_idx, view_count)
        num_views, num_objects, frame_count = visibility.shape
        
        all_objects_in_scene = list(range(num_objects))
        name_counts = self.data_manager.count_object_occurrences(scene_idx, all_objects_in_scene) 
        
        if any(count > 1 for count in name_counts.values()):
            print(f"Skipping {scene_idx}: Found duplicate objects.")
            return []
        
        qa_dataset_samples = []
        
        for event_obj in all_objects_in_scene:
            event_name = self.data_manager.get_object_name(scene_idx, event_obj)
            if name_counts[event_name] > 1: continue
                
            views_visible_in = [v for v in range(view_count) if visibility[v, event_obj, :].sum() > 0]
            if len(views_visible_in) == 0: continue
             
            event_cam_id = views_visible_in[0]
            vis_array = visibility[event_cam_id, event_obj, :]
            
            exits = np.where((vis_array[:-1] > 0) & (vis_array[1:] == 0))[0]
            if len(exits) == 0: continue
                
            exit_frame = exits[0] + 1
            if exit_frame >= frame_count or exit_frame < 12: continue 
                
            video_number = event_cam_id + 1
            
            for target_obj in all_objects_in_scene:
                if target_obj == event_obj: continue
                    
                target_name = self.data_manager.get_object_name(scene_idx, target_obj)
                if name_counts[target_name] > 1: continue
                    
                visible_in_all = all(visibility[v, target_obj, exit_frame] > 150 for v in range(view_count))
                if not visible_in_all: continue
                
                target_pos = np.array(self.data_manager.instance_data[scene_idx]['cam0'][target_obj]['positions'][exit_frame])
                
                distances = {}
                for other_obj in all_objects_in_scene:
                    if other_obj in (target_obj, event_obj): continue
                    
                    is_visible = any(visibility[v, other_obj, exit_frame] > 25 for v in range(view_count))
                    if not is_visible: continue
                        
                    other_pos = np.array(self.data_manager.instance_data[scene_idx]['cam0'][other_obj]['positions'][exit_frame])
                    dist = np.linalg.norm(target_pos - other_pos)
                    distances[other_obj] = dist
                    
                if not distances: continue 
                  
                sorted_by_distance = sorted(distances.items(), key=lambda item: item[1])
                closest_obj = sorted_by_distance[0][0]
                correct_answer = f"{self.data_manager.get_object_name(scene_idx, closest_obj)}"
                
                distractors = []
                for i in range(1, len(sorted_by_distance)):
                    if len(distractors) >= 3: break
                    candidate_name = self.data_manager.get_object_name(scene_idx, sorted_by_distance[i][0])
                    if candidate_name not in distractors and candidate_name != correct_answer and candidate_name != target_name:
                        distractors.append(candidate_name)
                    
                fake_adjectives = ["Huge Purple", "Tiny Black", "Large White", "Small Green"]
                fake_nouns = ["Sphere", "Cube", "Cylinder"]
                while len(distractors) < 3:
                    fake_name = f"{random.choice(fake_adjectives)} {random.choice(fake_nouns)}"
                    if fake_name not in distractors and fake_name not in (correct_answer, target_name):
                        distractors.append(fake_name)
                
                question = (
                    f"Observe the videos closely. At the exact moment the {event_name} completely exits the frame "
                    f"in Video {video_number}, which object is physically closest to the {target_name} in the 3D space?"
                )
                
                options = [correct_answer] + distractors
                random.shuffle(options)
                
                labels = ['A', 'B', 'C', 'D']
                options_list = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
                correct_label = next(opt for opt in options_list if opt.split(') ')[1] == correct_answer)
                    
                qa_dataset_samples.append({
                    'scene_idx': scene_idx,
                    'event_object': event_name,
                    'target_object': target_name,
                    'exit_frame': str(exit_frame),
                    'video_condition': f"Video {video_number}",
                    'video_list': video_list,
                    'question': question,
                    'options': options_list,
                    'answer': correct_label
                })
                break 
                
        bucket_early, bucket_mid, bucket_late = [], [], []
        for sample in qa_dataset_samples:
            if int(sample['exit_frame']) <= 10: bucket_early.append(sample)
            elif 10 < int(sample['exit_frame']) <= 20: bucket_mid.append(sample)
            else: bucket_late.append(sample)
                
        random.shuffle(bucket_early)
        random.shuffle(bucket_mid)
        
        final_samples = []
        if bucket_early: final_samples.extend(bucket_early[:1]) 
        if bucket_mid: final_samples.extend(bucket_mid[:2])   
        final_samples.extend(bucket_late) 
        
        return final_samples
        

def generate_spatial_measurement_dataset(path, total_num, save, output_path):  
    data_manager = KubricDataset(path=path)
    generator = SpatialMeasurementQAGenerator(data_manager=data_manager)
    qa_list = []
    
    for idx, scene_idx in enumerate(os.listdir(data_manager.path)):
        if not scene_idx.startswith("scene_"):
            continue
        
        scene_samples = generator.generate(scene_idx=scene_idx, view_count=2)
        qa_list.extend(scene_samples)
        print(f"{idx + 1}/{len(os.listdir(data_manager.path))} - Total samples collected: {len(qa_list)}")
        
        if len(qa_list) >= total_num:
            break
            
    if qa_list:
        print("Sample 0:", qa_list[0])  
    else:
        print("No samples were generated!")
     
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_jsonl_file(filepath=output_path, data_list=qa_list[:total_num])
        print(f"Successfully saved {len(qa_list[:total_num])} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kubric Spatial Measurement QA Dataset")
    
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the base directory containing Kubric scene folders."
    )
    
    parser.add_argument(
        "--total_num", 
        type=int, 
        default=1000, 
        help="Total number of QA samples to generate (default: 1000)."
    )
    
    parser.add_argument(
        "--save", 
        action="store_true", 
        help="Include this flag to save the output to a JSONL file."
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="json_files/Spatial_Tracking/Kubric_Spatial_Measurement.jsonl", 
        help="Path where the output JSONL file will be saved."
    )

    args = parser.parse_args()

    # Run the generator with parsed arguments
    generate_spatial_measurement_dataset(
        path=args.path, 
        total_num=args.total_num, 
        save=args.save,
        output_path=args.output_path
    )
ATTRIBUTE_DICT = {
    'color': {'blue': 0, 'gray': 1, 'green': 2, 'yellow':3, 'cyan':4, 'brown':5, 'purple':6, 'red':7},
    'material': {'rubber': 0, 'metal': 1},
    'shape': {'cube': 0, 'sphere': 1, 'cylinder': 2},
}

def get_abs_velocity(velocity_dict):
    return (velocity_dict[0]**2 + velocity_dict[1]**2 + velocity_dict[2]**2)**0.5

def get_fastest_speed(ann_dict):
    max_velocity, fastest_object, frame_idx = 0, None, None
    
    for item in ann_dict['motion_trajectory']:
        for object_dict in item['objects']:
            if not object_dict['inside_camera_view']:
                continue
            abs_velocity = get_abs_velocity(object_dict['velocity']) 
            if abs_velocity > max_velocity:
                max_velocity=abs_velocity
                fastest_object = ann_dict['object_property'][object_dict['object_id']]
                frame_idx = item['frame_id']
    
    return max_velocity, fastest_object, frame_idx
    
def get_top_k_fastest_objects(ann_dict, k=2):
    """
    Calculates velocities for all objects and returns the top k fastest object dictionaries.
    """
    object_velocities = list()
    
    for object_id in range(len(ann_dict['object_property'])):
        object_velocities.append({'object_name': get_object_name(ann_dict, object_id), 
                                  'max_vel':0})
        
    for item in ann_dict['motion_trajectory']:
        for object_idx, object_dict in enumerate(item['objects']):
            if not object_dict['inside_camera_view']:
                continue
            abs_velocity = get_abs_velocity(object_dict['velocity']) 
            if abs_velocity > object_velocities[object_idx]['max_vel']:
                object_velocities[object_idx]['max_vel'] = abs_velocity
                
    object_velocities.sort(key=lambda x: x['max_vel'], reverse=True)
    
    return object_velocities[:k]

def add_alphabet_to_options(options, answer):
    for opt_idx in range(len(options)):
        if options[opt_idx] == answer:
            answer = f"{chr(65 + opt_idx)}) {options[opt_idx]}"
        options[opt_idx] = f"{chr(65 + opt_idx)}) {options[opt_idx]}" 
    return options, answer

def get_aligned_frame_id(frame_id, nframes=32):
    frame_rate = 128 / nframes
    return int(frame_id // frame_rate) 

def get_time_value(frame_id, time_mode='frame', nframes=32, fps=25.0):
    """
    Converts the raw frame_id to either an aligned frame index or a timestamp.
    Calculates timestamp based on the provided FPS (default 25.0).
    """
    if time_mode == 'timestamp':
        return round(frame_id / fps, 1)
    return get_aligned_frame_id(frame_id, nframes)

def format_time_string(val, time_mode='frame'):
    """Formats the time value for display in options and reasoning."""
    if time_mode == 'timestamp':
        return f"{val:.1f}s"
    return f"Frame {val}"

def get_entrance_and_exit_frames(ann_dict):
    entrance_and_exit_dict = dict()
    
    for item in ann_dict['motion_trajectory']:
        for obj in item['objects']:
            obj_id = obj['object_id']
            if obj_id not in entrance_and_exit_dict:
                entrance_and_exit_dict[obj_id] = {'enter': None, 'exit': None}
            if obj['inside_camera_view'] and entrance_and_exit_dict[obj_id]['enter'] is None:
                entrance_and_exit_dict[obj_id]['enter'] = item['frame_id']
            elif not obj['inside_camera_view'] and entrance_and_exit_dict[obj_id]['exit'] is None and entrance_and_exit_dict[obj_id]['enter'] is not None:
                entrance_and_exit_dict[obj_id]['exit'] = item['frame_id']    
                
    return entrance_and_exit_dict


def get_collision_frame_dict(ann_dict):
    collision_frame_dict = dict()
    for item in ann_dict['collision']:
        frame_id = item['frame_id']

        collision_frame_dict[frame_id] = {'object_ids': item['object_ids'],
                                          'orientation_changes': [
                                              (ann_dict['motion_trajectory'][frame_id-1]['objects'][obj_id]['orientation'], 
                                               ann_dict['motion_trajectory'][frame_id]['objects'][obj_id]['orientation']) for obj_id in item['object_ids']]}
        

    return collision_frame_dict


def get_stationary_moving_status(ann_dict):
    stationary_moving_dict = dict()
    for item in ann_dict['motion_trajectory']:
        
        frame_id = item['frame_id']
        for obj in item['objects']:
            if not obj['inside_camera_view']:
                continue
            obj_id = obj['object_id']
            if sum(obj['angular_velocity']) == 0 and obj_id not in stationary_moving_dict:
                stationary_moving_dict[obj_id] = {'stationary': frame_id, 'moving': None}
                
            elif sum(obj['angular_velocity']) != 0 and obj_id in stationary_moving_dict and stationary_moving_dict[obj_id]['moving'] is None:
                stationary_moving_dict[obj_id]['moving'] = frame_id
    
    return stationary_moving_dict


def get_object_name(ann_dict, object_idx):
    object_dict = ann_dict['object_property'][object_idx]
    return f"{object_dict['color']} {object_dict['material']} {object_dict['shape']}" 
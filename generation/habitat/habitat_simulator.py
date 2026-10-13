"""
Habitat video renderer for SYNCR (Spatial Tracking / Holistic Synthesis tasks).

This supersedes the previous habitat_simulator.py, which used a weaker NavMesh
scan (5,000 random points + single center-pixel semantic lookup) and had no
path-validity checks. This version uses uniform grid
sampling over the NavMesh with majority-vote ("patch voting") semantic
assignment, plus degenerate-path handling in the spline fit. The core
simulation/graph/spline algorithm is unchanged from that source; only the
data root and scene-selection logic were made configurable so train and val
can be rendered with the same code instead of by editing constants in place.

Output layout (unchanged from the original, and from the shipped val data):
    <output_root>/general/<split>/<scene>/cam<i>/cam<i>.mp4
    <output_root>/general/<split>/<scene>/cam<i>/semantic_video_ids.npz
    <output_root>/general/<split>/<scene>/scene_semantic_objects.json
    <output_root>/route_plan/<split>/<scene>/route<k>/cam<i>/{cam<i>.mp4,semantic_video_ids.npz}
    <output_root>/route_plan/<split>/<scene>/route<k>/{scene_semantic_objects.json,video_routes.json}

This is exactly what generation/habitat/base_habitat.py's Habitat_Dataset expects via
root_path/section/split/scene[/route]/..., so no changes are needed downstream
in habitat_entity_tracking.py or habitat_holistic_aggregation.py.

Run:
    python -m generation.habitat.habitat_simulator --task route_plan --split train \
        --hm3d-root ${SYNCR_DATA}/habitat_orig_data/scene_datasets/hm3d \
        --output-root ${SYNCR_DATA}/habitat_data

    # Small-scale smoke test before a full run:
    python -m generation.habitat.habitat_simulator --task both --split train \
        --hm3d-root ./data/scene_datasets/hm3d --output-root /tmp/habitat_smoke \
        --limit-scenes 2
"""

import argparse
import habitat_sim
import numpy as np
import random
import os
import json
from scipy.interpolate import splprep, splev
import math
import imageio
import networkx as nx

# ---------------------------------------------------------------------
# Configuration: Static Parameters
# ---------------------------------------------------------------------
class Config:
    # Simulator Settings
    RESOLUTION = [720, 1280]
    # Fallback HM3D root when --hm3d-root is not passed. Only meaningful when
    # the process is run with this as its cwd; prefer --hm3d-root explicitly.
    DATASET_ROOT = 'data/scene_datasets/hm3d'

    # Video Generation Settings
    FPS = 5
    NUM_FRAMES = 81 # Yields 80 walking frames + 20 pan frames = 100 frames total
    STEP_SIZE = 0.1

    # Camera Smoothing & Pan Settings
    LOOK_AHEAD_FRAMES = 10
    PAN_SECONDS = 4
    PAN_SWEEP_ANGLE = math.radians(240)
    MAX_TILT = math.radians(40)

    # Graph & Routing Thresholds
    MAX_PATH_NODES = 6
    GRID_STEP_SIZE = 0.25 # Meters between samples for the NavMesh grid

    # Random Video Settings
    NUM_STEPS = 200
    DISTANCE_RANGE = [15.0, 20.0]

# ---------------------------------------------------------------------
# Static Helpers
# ---------------------------------------------------------------------
def get_geodesic_distance(pathfinder, start_pos, end_pos):
    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = end_pos

    found_path = pathfinder.find_path(path)
    if found_path:
        return path.geodesic_distance
    return np.inf

def load_semantic_mapping(txt_filepath):
    mapping = {}
    with open(txt_filepath, 'r') as f:
        for line in f:
            clean_line = line.replace('"', '').replace(',', ' ').strip()
            if not clean_line:
                continue
            parts = clean_line.split()
            try:
                obj_id = int(parts[0])
                obj_name = " ".join(parts[1:]).lower()
                mapping[obj_id] = obj_name
            except ValueError:
                pass
    print(f"Loaded {len(mapping)} semantic object definitions from {txt_filepath}.")
    return mapping

def load_object_to_region_mapping(scene_path):
    mapping = {}
    txt_path = [f for f in os.listdir(scene_path) if f.endswith('.semantic.txt')][0]
    with open(os.path.join(scene_path, txt_path), 'r') as f:
        for line in f:
            clean_line = line.replace('"', '').replace(',', ' ').strip()
            if not clean_line: continue
            parts = clean_line.split()
            try:
                obj_id = int(parts[0])
                region_id = parts[-1]
                mapping[obj_id] = region_id
            except ValueError:
                pass
    return mapping

# ---------------------------------------------------------------------
# Core Generator Class
# ---------------------------------------------------------------------
class HabitatVideoGenerator:
    def __init__(self, split_type, hm3d_root=None, output_root='.'):
        self.split = split_type
        self.hm3d_root = hm3d_root or Config.DATASET_ROOT
        self.base_path = os.path.join(self.hm3d_root, self.split)
        # Where general/<split>/... and route_plan/<split>/... get written.
        # Kept separate from hm3d_root: the original code wrote these relative
        # to the process's cwd, which only worked by an implicit convention of
        # running from inside the intended output directory.
        self.output_root = output_root
        self.sim = None
        self.semantic_dict = {}
        self.current_sim_id = None

    def cleanup(self):
        """Safely closes the simulator to free up OpenGL resources."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            print("Simulator closed.")

    # --- Setup & Configuration ---

    def _make_cfg(self, scene_filepath, dataset_config_path):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_filepath
        sim_cfg.scene_dataset_config_file = dataset_config_path

        sensors = []
        for uuid, sensor_type in [("color_sensor", habitat_sim.SensorType.COLOR),
                                  ("depth_sensor", habitat_sim.SensorType.DEPTH),
                                  ("semantic_sensor", habitat_sim.SensorType.SEMANTIC)]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            spec.resolution = Config.RESOLUTION
            sensors.append(spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def load_sim(self, sim_id):
        if self.current_sim_id == sim_id:
            return

        short_id = sim_id.split('-')[-1] if '-' in sim_id else sim_id
        scene_dir = os.path.join(self.base_path, sim_id)

        semantic_txt_file = os.path.join(scene_dir, f"{short_id}.semantic.txt")
        scene_file = os.path.join(scene_dir, f"{short_id}.basis.glb")
        dataset_config_file = os.path.join(self.base_path, f"hm3d_annotated_{self.split}_basis.scene_dataset_config.json")
        navmesh_file = os.path.join(scene_dir, f"{short_id}.basis.navmesh")

        cfg = self._make_cfg(scene_file, dataset_config_file)

        if self.sim is None:
            print(f"Initializing new Simulator instance for {short_id}...")
            self.sim = habitat_sim.Simulator(cfg)
        else:
            print(f"Reconfiguring existing Simulator instance for {short_id}...")
            self.sim.reconfigure(cfg)

        self.sim.pathfinder.load_nav_mesh(navmesh_file)
        if not self.sim.pathfinder.is_loaded:
            print(f"Warning: The .navmesh file could not be loaded for {sim_id}!")

        self.semantic_dict = load_semantic_mapping(semantic_txt_file)
        self.current_sim_id = sim_id

    # --- Core Video Generation ---

    def generate_smooth_video_clip(self, start_pos, start_rot, target_pos=None):
        if target_pos is None:
            target_pos = self.sim.pathfinder.get_random_navigable_point()

        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = target_pos
        found_path = self.sim.pathfinder.find_path(path)

        if not found_path or len(path.points) == 0:
            print("  -> Warning: Pathfinder failed to connect these specific points.")
            return []

        def densify_waypoints_spline(sparse_waypoints):
            # 1. Remove consecutive duplicates which can crash the spline math
            unique_pts = [sparse_waypoints[0]]
            for pt in sparse_waypoints[1:]:
                if np.linalg.norm(np.array(pt) - np.array(unique_pts[-1])) > 1e-4:
                    unique_pts.append(pt)

            pts = np.array(unique_pts)
            dense_waypoints = []

            # 2. If the path is literally just standing still, duplicate the frame
            if len(pts) < 2:
                for _ in range(Config.NUM_FRAMES):
                    dense_waypoints.append(self.sim.pathfinder.snap_point(pts[0]))
                return dense_waypoints

            # 3. Dynamically set spline degree 'k'
            # Default is 3 (cubic curve), but if we only have 2 points, use 1 (straight line)
            k = min(3, len(pts) - 1)

            tck, u = splprep([pts[:,0], pts[:,1], pts[:,2]], s=0.5, k=k)
            u_new = np.linspace(0, 1, Config.NUM_FRAMES)
            smooth_x, smooth_y, smooth_z = splev(u_new, tck)

            for i in range(Config.NUM_FRAMES):
                pt = np.array([smooth_x[i], smooth_y[i], smooth_z[i]])
                snapped_pt = self.sim.pathfinder.snap_point(pt)
                if not np.isnan(snapped_pt[0]):
                    dense_waypoints.append(snapped_pt)
                else:
                    dense_waypoints.append(pt) # Fallback
            return dense_waypoints

        waypoints = densify_waypoints_spline(path.points)
        frames = []
        yaw = 0.0

        # Walk forward
        for i in range(len(waypoints)-1):
            current_pos = waypoints[i]

            look_ahead_idx = min(i + Config.LOOK_AHEAD_FRAMES, len(waypoints) - 1)
            target_look_pos = waypoints[look_ahead_idx]
            direction = target_look_pos - current_pos

            if np.linalg.norm(direction) > 1e-5:
                yaw = math.atan2(direction[0], direction[2])

            rotation = [0.0, np.sin(yaw / 2.0), 0.0, np.cos(yaw / 2.0)]
            self.sim.agents[0].set_state(habitat_sim.AgentState(current_pos, rotation))
            frames.append(self._get_frame_data(current_pos, rotation))

        # Final Pan Sweep (Looking DOWN)
        final_pos = waypoints[-1]
        pan_frames = Config.FPS * Config.PAN_SECONDS

        for step in range(pan_frames):
            pan_yaw = yaw + (step * (Config.PAN_SWEEP_ANGLE / pan_frames))
            pan_pitch = -math.sin((step / pan_frames) * math.pi) * Config.MAX_TILT

            qx = math.sin(pan_pitch / 2.0) * math.cos(pan_yaw / 2.0)
            qy = math.cos(pan_pitch / 2.0) * math.sin(pan_yaw / 2.0)
            qz = -math.sin(pan_pitch / 2.0) * math.sin(pan_yaw / 2.0)
            qw = math.cos(pan_pitch / 2.0) * math.cos(pan_yaw / 2.0)
            pan_rot = [qx, qy, qz, qw]

            self.sim.agents[0].set_state(habitat_sim.AgentState(final_pos, pan_rot))
            frames.append(self._get_frame_data(final_pos, pan_rot))

        return frames

    # --- Graph & Routing Logic ---

    def _build_ground_truth_apartment_graph(self, scene_idx):
        """Builds a topological graph using robust grid sampling and pathfinding validation."""
        print(f"\nScanning the physical NavMesh for {scene_idx} using grid sampling...")
        scene_path = os.path.join(self.base_path, scene_idx)
        object_to_region = load_object_to_region_mapping(scene_path)

        look_down_rot = [-0.7071068, 0.0, 0.0, 0.7071068]

        # 1. Uniform Grid Sampling across the NavMesh
        bounds = self.sim.pathfinder.get_bounds()
        min_b, max_b = bounds[0], bounds[1]

        valid_points = []
        for x in np.arange(min_b[0], max_b[0], Config.GRID_STEP_SIZE):
            for z in np.arange(min_b[2], max_b[2], Config.GRID_STEP_SIZE):
                pt = np.array([x, min_b[1], z])
                snapped_pt = self.sim.pathfinder.snap_point(pt)
                # Only keep the point if it successfully snapped to the NavMesh
                if not np.isnan(snapped_pt[0]):
                    valid_points.append(snapped_pt)

        point_to_region = {}
        region_to_points = {}

        # 2. Robust Semantic Checking (Patch Voting)
        for pt in valid_points:
            self.sim.agents[0].set_state(habitat_sim.AgentState(pt, look_down_rot))
            obs = self.sim.get_sensor_observations()

            center_y, center_x = obs["semantic_sensor"].shape[0] // 2, obs["semantic_sensor"].shape[1] // 2
            patch_size = 5 # 10x10 pixel patch

            # Extract a small patch from the center of the camera
            semantic_patch = obs["semantic_sensor"][
                max(0, center_y - patch_size) : min(obs["semantic_sensor"].shape[0], center_y + patch_size),
                max(0, center_x - patch_size) : min(obs["semantic_sensor"].shape[1], center_x + patch_size)
            ]

            # Get the majority vote (mode) of semantic IDs in the patch
            flattened = semantic_patch.flatten()
            if len(flattened) == 0: continue

            values, counts = np.unique(flattened, return_counts=True)
            floor_obj_id = values[np.argmax(counts)]

            if floor_obj_id in object_to_region:
                region_id = object_to_region[floor_obj_id]
                pt_tuple = tuple(pt)
                point_to_region[pt_tuple] = region_id

                if region_id not in region_to_points:
                    region_to_points[region_id] = []
                region_to_points[region_id].append(pt_tuple)

        G = nx.Graph()
        regions = list(region_to_points.keys())
        for r_id in regions:
            G.add_node(r_id)

        # 3. Continuous Traversal Check for Edges
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1 = regions[i]
                r2 = regions[j]

                connected = False
                # Look for a "doorway" by finding points in r1 and r2 that are physically close
                for p1 in region_to_points[r1]:
                    for p2 in region_to_points[r2]:
                        # If points are physically close (< 1.5m)
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < 1.5:

                            # Check if they are also navigably close (not blocked by a thin wall)
                            path = habitat_sim.ShortestPath()
                            path.requested_start = p1
                            path.requested_end = p2
                            found_path = self.sim.pathfinder.find_path(path)

                            if found_path and path.geodesic_distance < 2.0:
                                G.add_edge(r1, r2)
                                connected = True
                                break # Found a valid doorway, stop checking points for this pair
                    if connected:
                        break # Move on to the next region pair

        return G, point_to_region

    def _find_paths_in_range(self, G, low, hi):
        """Finds all unique shortest paths whose node count falls within [low, hi]."""
        valid_paths = []
        nodes = list(G.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source = nodes[i]
                target = nodes[j]

                # Check if they are connected at all
                if not nx.has_path(G, source, target):
                    continue

                # Find the single shortest path
                shortest = nx.shortest_path(G, source, target)

                # Check if its length (number of nodes) is in the acceptable range
                if low <= len(shortest) <= hi:
                    valid_paths.append({
                        "start_region": source,
                        "end_region": target,
                        "path": shortest
                    })

        print(f"Found {len(valid_paths)} valid paths with length between {low} and {hi} nodes.")
        return valid_paths

    def _plan_path_tour(self, scenario, G):
        """Splits a path into 3 segments, exploring a neighbor for short paths."""
        path = scenario["path"]
        video_plans = []

        num_edges = len(path) - 1

        if num_edges < 2:
            return []

        # --- NEW NEIGHBOR EXPLORATION FOR 2-EDGE PATHS ---
        if num_edges == 2:
            # We have a path like [4, 5, 6]
            room_A, room_B, room_C = path[0], path[1], path[2]

            # Find all connected rooms to Room C (Region 6)
            all_neighbors = list(G.neighbors(room_C))

            # Try to find a neighbor that ISN'T Room B (so we don't backtrack)
            forward_neighbors = [n for n in all_neighbors if n != room_B]

            if forward_neighbors:
                # Pick a random new room to explore!
                room_D = random.choice(forward_neighbors)
            else:
                # Room C is a dead end. We have no choice but to backtrack.
                room_D = room_B

            video_plans.append({"start": room_A, "end": room_B, "sub_path": [room_A, room_B]})
            video_plans.append({"start": room_B, "end": room_C, "sub_path": [room_B, room_C]})
            video_plans.append({"start": room_C, "end": room_D, "sub_path": [room_C, room_D]})

            return video_plans

        # --- STANDARD CHUNKING FOR >= 3 EDGES ---
        base_size = num_edges // 3
        remainder = num_edges % 3

        chunk_sizes = [base_size] * 3
        for i in range(remainder):
            chunk_sizes[i] += 1

        current_idx = 0
        for size in chunk_sizes:
            sub_path = path[current_idx : current_idx + size + 1]
            start_region = sub_path[0]
            end_region = sub_path[-1]

            video_plans.append({
                "start": start_region,
                "end": end_region,
                "sub_path": sub_path
            })
            current_idx += size

        return video_plans

    def _get_anchored_waypoints(self, r_start, r_end, point_map):
        pts_start = [pt for pt, r_id in point_map.items() if r_id == str(r_start)]
        pts_end = [pt for pt, r_id in point_map.items() if r_id == str(r_end)]

        if not pts_start or not pts_end:
            return None, None

        def get_deep_point(target_pts, reference_pt):
            sorted_pts = sorted(target_pts, key=lambda p: np.linalg.norm(np.array(p) - np.array(reference_pt)), reverse=True)
            top_tier = max(1, len(sorted_pts) // 10)
            return random.choice(sorted_pts[:top_tier])

        temp_end = random.choice(pts_end)
        start_pos = get_deep_point(pts_start, temp_end)
        end_pos = get_deep_point(pts_end, start_pos)

        return start_pos, end_pos

    # --- Data Saving Helpers ---

    def _get_frame_data(self, pos, rot):
        obs = self.sim.get_sensor_observations()
        return {
            "rgb": obs["color_sensor"],
            "depth": obs["depth_sensor"],
            "semantic": obs["semantic_sensor"],
            "position": pos,
            "rotation": rot
        }

    def _save_video_frames(self, frames, output_base_dir, video_name):
        video_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)

        video_unique_ids = set()
        all_rgb_frames = []
        all_semantic_frames = []

        for frame_data in frames:
            all_rgb_frames.append(frame_data["rgb"][:, :, :3])
            semantic_array = frame_data["semantic"]
            all_semantic_frames.append(semantic_array)
            video_unique_ids.update(np.unique(semantic_array).tolist())

        imageio.mimwrite(os.path.join(video_dir, f"{video_name}.mp4"), all_rgb_frames, fps=Config.FPS, macro_block_size=None)
        np.savez_compressed(os.path.join(video_dir, "semantic_video_ids.npz"), semantics=np.array(all_semantic_frames, dtype=np.int16))

        print(f"Saved {video_name}.mp4 with {len(frames)} frames.")
        return video_unique_ids

    def _save_semantic_dict(self, dataset_folder, all_scene_ids):
        scene_semantic_mapping = {}
        for uid in all_scene_ids:
            if uid == 0:
                scene_semantic_mapping[uid] = "background/unlabeled"
            else:
                scene_semantic_mapping[uid] = self.semantic_dict.get(uid, "unknown_object")

        with open(os.path.join(dataset_folder, "scene_semantic_objects.json"), "w") as f:
            json.dump(scene_semantic_mapping, f, indent=4)

    # --- High-Level Pipelines ---

    def generate_route_plan_videos(self, limit_scenes=None):
        route_plan_root = os.path.join(self.output_root, "route_plan", self.split)
        scene_ids = sorted(s for s in os.listdir(self.base_path) if '-' in s)
        if limit_scenes is not None:
            scene_ids = scene_ids[:limit_scenes]

        for sim_id in scene_ids:
            short_sim_id = sim_id.split('-')[1]
            if not os.path.exists(os.path.join(self.base_path, sim_id, f"{short_sim_id}.semantic.txt")) \
                or os.path.exists(os.path.join(route_plan_root, short_sim_id)):
                continue

            self.load_sim(sim_id)
            G, point_map = self._build_ground_truth_apartment_graph(sim_id)

            # We need at least 4 nodes to generate 3 separate video steps (3 edges)
            path_scenarios = self._find_paths_in_range(G, low=3, hi=float('inf'))

            if not path_scenarios:
                print("No paths of suitable length found in this apartment. Skipping.")
                continue

            for route_idx, scenario in enumerate(path_scenarios):
                try:
                    self._process_route_scenario(scenario, point_map, short_sim_id, route_idx, G, route_plan_root)
                    print(scenario)
                except Exception as e:
                    print(f"Failed on {sim_id}, route {route_idx}: {e}")


    def _process_route_scenario(self, scenario, point_map, short_sim_id, route_idx, G, route_plan_root):
        camera_plans = self._plan_path_tour(scenario, G)

        if not camera_plans:
            print(f"  -> Path too short to split, skipping route {route_idx}.")
            return

        dataset_folder = os.path.join(route_plan_root, short_sim_id, f"route{route_idx}")
        generated_videos = list()
        baton_pos = None

        for idx, plan in enumerate(camera_plans):
            r_start, r_end = plan['start'], plan['end']
            sub_path = plan['sub_path'] # Get the new sub_path
            cam_name = f"cam{idx}"

            print(f"Generating {cam_name}: Path {sub_path}")

            temp_start, target_pos = self._get_anchored_waypoints(r_start, r_end, point_map)

            if temp_start is None or target_pos is None:
                print(f"  -> Failed to find valid waypoints for {cam_name}. Aborting route.")
                return

            if baton_pos is not None:
                start_pos = baton_pos
            else:
                start_pos = temp_start

            yaw = math.atan2(target_pos[0] - start_pos[0], target_pos[2] - start_pos[2])
            start_rot = [0.0, np.sin(yaw / 2.0), 0.0, np.cos(yaw / 2.0)]

            vid_frames = self.generate_smooth_video_clip(start_pos, start_rot, target_pos)

            if not vid_frames:
                print(f"  -> Skipping {cam_name} due to empty video frames.")
                return

            baton_pos = target_pos

            generated_videos.append({
                "cam_name": cam_name,
                "frames": vid_frames,
                "route_str": f"{r_start} -> {r_end}",
                "full_path": sub_path # Store it here
            })

        if len(generated_videos) != len(camera_plans):
            print(f"  -> Did not generate all required videos. Aborting route {route_idx}.")
            return

        print(f"All {len(camera_plans)} cameras generated successfully! Saving to {dataset_folder}...")
        os.makedirs(dataset_folder, exist_ok=True)

        routes_dict = {}
        all_scene_ids = set()

        for video_data in generated_videos:
            cam_name = video_data["cam_name"]
            vid_frames = video_data["frames"]

            # Upgrade the JSON dictionary to hold the complex object
            routes_dict[cam_name] = video_data["full_path"]

            ids_vid = self._save_video_frames(vid_frames, dataset_folder, cam_name)
            all_scene_ids |= ids_vid

        self._save_semantic_dict(dataset_folder, all_scene_ids)

        with open(os.path.join(dataset_folder, "video_routes.json"), "w") as f:
            json.dump(routes_dict, f, indent=4)


    def generate_random_videos(self, limit_scenes=None, max_target_retries=5):
        general_root = os.path.join(self.output_root, "general", self.split)
        problematic_scenes = set()

        scene_ids = sorted(s for s in os.listdir(self.base_path) if '-' in s)
        if limit_scenes is not None:
            scene_ids = scene_ids[:limit_scenes]

        for sim_id in scene_ids:
            short_sim_id = sim_id.split('-')[1]

            dataset_folder = os.path.join(general_root, short_sim_id)
            if os.path.exists(dataset_folder):
                continue

            # Only ~18% of HM3D scenes ship semantic annotations. load_sim()
            # loads the full mesh into the simulator before it would discover
            # semantic.txt is missing, so checking here first (as
            # generate_route_plan_videos already does) avoids paying that cost
            # for every unannotated scene -- the difference between this
            # finishing in minutes and hours on a full sweep.
            if not os.path.exists(os.path.join(self.base_path, sim_id, f"{short_sim_id}.semantic.txt")):
                continue

            try:
                self.load_sim(sim_id)
                pathfinder = self.sim.pathfinder

                candidates = [pathfinder.get_random_navigable_point() for _ in range(200)]
                start_a = candidates[0]

                max_dist_b = -1
                start_b = candidates[1]
                for pt in candidates:
                    dist = get_geodesic_distance(pathfinder, start_a, pt)
                    if dist != np.inf and dist > max_dist_b:
                        max_dist_b, start_b = dist, pt

                start_c = candidates[2]

                def get_local_target(start_pos):
                    for _ in range(Config.NUM_STEPS):
                        cand = pathfinder.get_random_navigable_point()
                        dist = get_geodesic_distance(pathfinder, start_pos, cand)
                        if dist != np.inf and Config.DISTANCE_RANGE[0] < dist < Config.DISTANCE_RANGE[1]:
                            return cand
                    return pathfinder.get_random_navigable_point()

                starts = [start_a, start_b, start_c]

                # get_local_target's own fallback (a plain random navigable point,
                # used when no point in the target distance range is found) can
                # land in a disconnected NavMesh island that the pathfinder can
                # never reach from `start_pos` -- generate_smooth_video_clip then
                # returns an empty frame list. Retrying with a fresh target
                # resolves this in practice; if it never does, the camera (and
                # so the whole scene, since ReID/Counting need all views valid)
                # is skipped rather than silently saving a 0-frame video whose
                # .mp4 is never even written by imageio, while its path would
                # still be added to video_path_list by callers.
                all_video_data = []
                for i, st in enumerate(starts):
                    frames = []
                    for attempt in range(max_target_retries):
                        tg = get_local_target(st)
                        angle = random.uniform(0, 2 * np.pi)
                        rot = [0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)]
                        print(f"Generating Video {i} (attempt {attempt + 1}/{max_target_retries})...")
                        frames = self.generate_smooth_video_clip(st, rot, target_pos=tg)
                        if frames:
                            break

                    if not frames:
                        print(f"  -> cam{i} never found a reachable target after "
                              f"{max_target_retries} attempts. Aborting scene {sim_id}.")
                        all_video_data = []
                        break

                    all_video_data.append((f"cam{i}", frames))

                if len(all_video_data) != len(starts):
                    problematic_scenes.add(sim_id)
                    continue

                os.makedirs(dataset_folder, exist_ok=True)
                all_scene_ids = set()
                for cam_name, frames in all_video_data:
                    ids = self._save_video_frames(frames, dataset_folder, cam_name)
                    all_scene_ids |= ids

                self._save_semantic_dict(dataset_folder, all_scene_ids)

            except Exception as e:
                print(f"Failed on {sim_id} with error: {e}")
                problematic_scenes.add(sim_id)

        if problematic_scenes:
            print(f"Scenes that raised errors: {sorted(problematic_scenes)}")

# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    parser = argparse.ArgumentParser(
        description="Render Habitat multi-camera videos for SYNCR.")

    parser.add_argument("--split", type=str, default='val', choices=['train', 'val'],
                        help="HM3D split to render from.")
    parser.add_argument("--task", type=str, default='both',
                        choices=['route_plan', 'random', 'both'],
                        help="'random' -> general/ (feeds ReID, Counting). "
                             "'route_plan' -> route_plan/ (feeds Counting, Route Planning). "
                             "'both' runs both.")
    parser.add_argument("--hm3d-root", type=str, default=Config.DATASET_ROOT,
                        help="Root of the raw HM3D scene assets "
                             "(contains train/, val/, hm3d_*_basis.scene_dataset_config.json).")
    parser.add_argument("--output-root", type=str, default='.',
                        help="Root under which general/<split>/ and route_plan/<split>/ "
                             "are created. Point this at the same 'habitat_data' root "
                             "already used for the val benchmark.")
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="Process only the first N eligible scenes. For a small-scale "
                             "smoke test before committing to a full render.")

    args = parser.parse_args()

    generator = HabitatVideoGenerator(
        split_type=args.split,
        hm3d_root=args.hm3d_root,
        output_root=args.output_root,
    )
    try:
        if args.task in ('route_plan', 'both'):
            generator.generate_route_plan_videos(limit_scenes=args.limit_scenes)
        if args.task in ('random', 'both'):
            generator.generate_random_videos(limit_scenes=args.limit_scenes)
    finally:
        generator.cleanup()

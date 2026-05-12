"""
Unified data generation script for SYNCR datasets.

This script consolidates argument parsing for all data generation tasks across
different datasets (Kubric, CLEVRER, Habitat) into a single entry point.

Supports all 7 SYNCR benchmark tasks across 4 cognitive pillars:

  Temporal Alignment:
    - kubric_sync: Multi-angle temporal synchronization
    - clevrer_temporal: Sequential ordering of video segments

  Spatial Tracking:
    - kubric_spatial: Spatial measurement and tracking
    - habitat_tracking: Object re-identification across viewpoints

  Comparative Reasoning:
    - clevrer_comparative: Kinematic and collision comparison

  Holistic Synthesis:
    - habitat_holistic: Object counting and route planning

Usage:
    python generate_data.py --dataset kubric_sync --path /path/to/kubric --total-num 1000
    python generate_data.py --dataset clevrer_temporal --root-path /path/to/clevrer --total-num 1000
    python generate_data.py --dataset clevrer_comparative --root-path /path/to/clevrer --task-type kinematic
    python generate_data.py --dataset habitat_holistic --root-path /path/to/habitat --task object_counting
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments for data generation."""
    parser = argparse.ArgumentParser(
        description="Unified data generation script for SYNCR datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Primary argument: which dataset to generate
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "kubric_sync",
            "clevrer_temporal",
            "habitat_tracking",
            "kubric_spatial",
            "clevrer_comparative",
            
            "habitat_holistic",
        ],
        help="Which dataset to generate data for.",
    )
    
    # ========== Common Arguments ==========
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the output to a JSONL file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose terminal output.",
    )
    
    # ========== Kubric Arguments ==========
    parser.add_argument(
        "--path",
        type=str,
        help="[Kubric] Path to the base directory containing Kubric scene folders.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="[Kubric] Path where the output JSONL file will be saved.",
    )
    
    # ========== CLEVRER Arguments ==========
    parser.add_argument(
        "--root-path",
        type=str,
        help="[CLEVRER/Habitat] Root path for the dataset.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["collision", "kinematic"],
        help="[CLEVRER] Task type to generate (collision or kinematic).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="[CLEVRER] Number of videos/objects to compare.",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=32,
        help="[CLEVRER] Number of frames for collision data.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        choices=["timestamp", "frame"],
        default="timestamp",
        help="[CLEVRER] Time mode setting for collision tracking.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="[CLEVRER] Threshold for kinematic max velocity comparison.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="[CLEVRER] Split of the dataset to use (e.g., validation, train, test).",
    )
    
    # ========== CLEVRER Temporal Arguments ==========
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="[CLEVRER Temporal] Number of video segments to sequence.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=128,
        help="[CLEVRER Temporal] Total frames for sequence generation.",
    )
    parser.add_argument(
        "--gap-time",
        type=float,
        default=0.0,
        help="[CLEVRER Temporal] Time gap (in seconds) between chronological video segments.",
    )
    
    # ========== Habitat Arguments ==========
    parser.add_argument(
        "--task",
        type=str,
        choices=["object_counting", "route_plan"],
        help="[Habitat] Select which generation task to run.",
    )
    parser.add_argument(
        "--route-mode",
        type=str,
        choices=["all", "longest", "medium"],
        default="medium",
        help="[Habitat] Mode for determining path length targets in route_plan task.",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="general",
        help="[Habitat] Data section to initialize Habitat dataset with.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="[Habitat Tracking] Frames per second for time conversions.",
    )
    parser.add_argument(
        "--min-range-len",
        type=int,
        default=3,
        help="[Habitat Tracking] Minimum frames a single continuous range must have.",
    )
    parser.add_argument(
        "--min-total-frames",
        type=int,
        default=5,
        help="[Habitat Tracking] Minimum total frames the object must be visible.",
    )
    
    # ========== Generic Arguments ==========
    parser.add_argument(
        "--total-num",
        type=int,
        default=None,
        help="Total number of samples to generate (default varies by dataset).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="[CLEVRER] Number of comparative QA samples to generate.",
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate that required arguments for the selected dataset are provided."""
    errors = []
    
    if args.dataset.startswith("kubric"):
        if not args.path:
            errors.append("--path is required for Kubric datasets")
        if args.total_num is None:
            args.total_num = 1000
    
    elif args.dataset == "clevrer_temporal":
        if not args.root_path:
            errors.append("--root-path is required for CLEVRER datasets")
        if args.total_num is None:
            args.total_num = 1000
    
    elif args.dataset == "clevrer_comparative":
        if not args.root_path:
            errors.append("--root-path is required for CLEVRER datasets")
        if not args.task_type:
            errors.append("--task-type is required for CLEVRER datasets")
        if args.total_num is None:
            args.total_num = 1000
    
    elif args.dataset.startswith("habitat"):
        if not args.root_path:
            errors.append("--root-path is required for Habitat datasets")
        if args.dataset == "habitat_holistic" and not args.task:
            errors.append("--task is required for habitat_holistic dataset")
        if args.total_num is None:
            args.total_num = 10000
    
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def generate_kubric_sync(args):
    """Generate Kubric Multi-Angle Synchronization dataset."""
    from kubric.kubric_sync import generate_multi_angle_sync_dataset
    
    print(f"Generating Kubric Multi-Angle Synchronization QA dataset...")
    print(f"  Path: {args.path}")
    print(f"  Total samples: {args.total_num}")
    print(f"  Save: {args.save}")
    if args.output_path:
        print(f"  Output path: {args.output_path}")
    
    output_path = args.output_path or "json_files/Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl"
    
    generate_multi_angle_sync_dataset(
        path=args.path,
        total_num=args.total_num,
        save=args.save,
        output_path=output_path,
    )


def generate_kubric_spatial(args):
    """Generate Kubric Spatial Measurement dataset."""
    from kubric.kubirc_spatial_meas import generate_spatial_measurement_dataset
    
    print(f"Generating Kubric Spatial Measurement QA dataset...")
    print(f"  Path: {args.path}")
    print(f"  Total samples: {args.total_num}")
    print(f"  Save: {args.save}")
    if args.output_path:
        print(f"  Output path: {args.output_path}")
    
    output_path = args.output_path or "json_files/Spatial_Tracking/Kubric_Spatial_Measurement.jsonl"
    
    generate_spatial_measurement_dataset(
        path=args.path,
        total_num=args.total_num,
        save=args.save,
        output_path=output_path,
    )


def generate_clevrer_temporal(args):
    """Generate CLEVRER Temporal Alignment (Sequential Ordering) dataset."""
    from clevrer.clvr_temporal_alignment import main as clevrer_temporal_main
    
    print(f"Generating CLEVRER Temporal Alignment (Sequential Ordering) QA dataset...")
    print(f"  Root path: {args.root_path}")
    print(f"  Split: {args.split}")
    print(f"  Total samples: {args.total_num}")
    print(f"  K (segments): {args.k}")
    print(f"  Total frames: {args.total_frames}")
    print(f"  Gap time: {args.gap_time}s")
    print(f"  Save: {args.save}")
    
    # Create mock sys.argv for CLEVRER Temporal's argparse
    sys_argv_backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--root-path", args.root_path,
        "--total-num", str(args.total_num),
        "--split", args.split,
        "--k", str(args.k),
        "--total-frames", str(args.total_frames),
        "--gap-time", str(args.gap_time),
    ]
    
    if args.save:
        sys.argv.append("--save")
    
    try:
        clevrer_temporal_main()
    finally:
        sys.argv = sys_argv_backup


def generate_clevrer_comparative(args):
    """Generate CLEVRER Comparative Reasoning dataset."""
    from clevrer.clvr_comparative_reasoning import main as clevrer_main
    
    print(f"Generating CLEVRER Comparative Reasoning QA dataset...")
    print(f"  Root path: {args.root_path}")
    print(f"  Task type: {args.task_type}")
    print(f"  Split: {args.split}")
    print(f"  Total samples: {args.total_num}")
    print(f"  Num samples: {args.num_samples}")
    
    # Create mock sys.argv for CLEVRER's argparse
    sys_argv_backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--root-path", args.root_path,
        "--num-samples", str(args.num_samples),
        "--total-num", str(args.total_num),
        "--task-type", args.task_type,
        "--split", args.split,
        "--k", str(args.k),
        "--nframes", str(args.nframes),
        "--time-mode", args.time_mode,
        "--threshold", str(args.threshold),
    ]
    
    try:
        clevrer_main()
    finally:
        sys.argv = sys_argv_backup


def generate_habitat_holistic(args):
    """Generate Habitat Holistic Aggregation dataset."""
    from habitat.habitat_holistic_aggregation import main as habitat_main
    
    print(f"Generating Habitat Holistic Aggregation QA dataset...")
    print(f"  Root path: {args.root_path}")
    print(f"  Task: {args.task}")
    print(f"  Route mode: {args.route_mode}")
    print(f"  Section: {args.section}")
    print(f"  Total samples: {args.total_num}")
    print(f"  Save: {args.save}")
    print(f"  Verbose: {args.verbose}")
    
    # Create mock sys.argv for Habitat's argparse
    sys_argv_backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--root-path", args.root_path,
        "--task", args.task,
        "--route-mode", args.route_mode,
        "--total-num", str(args.total_num),
        "--section", args.section,
    ]
    
    if args.save:
        sys.argv.append("--save")
    if args.verbose:
        sys.argv.append("--verbose")
    
    try:
        habitat_main()
    finally:
        sys.argv = sys_argv_backup


def generate_habitat_tracking(args):
    """Generate Habitat Object Re-identification (Tracking) dataset."""
    from habitat.habitat_spatial_tracking import main as habitat_tracking_main
    
    print(f"Generating Habitat Object Re-identification QA dataset...")
    print(f"  Root path: {args.root_path}")
    print(f"  FPS: {args.fps}")
    print(f"  Min range length: {args.min_range_len}")
    print(f"  Min total frames: {args.min_total_frames}")
    print(f"  Save: {args.save}")
    print(f"  Verbose: {args.verbose}")
    
    # Create mock sys.argv for Habitat Tracking's argparse
    sys_argv_backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--root-path", args.root_path,
        "--fps", str(args.fps),
        "--min-range-len", str(args.min_range_len),
        "--min-total-frames", str(args.min_total_frames),
    ]
    
    if args.save:
        sys.argv.append("--save")
    if args.verbose:
        sys.argv.append("--verbose")
    
    try:
        habitat_tracking_main()
    finally:
        sys.argv = sys_argv_backup


def main():
    """Main entry point for data generation."""
    args = parse_args()
    validate_args(args)
    
    print("\n" + "="*70)
    print("SYNCR Unified Data Generation")
    print("="*70 + "\n")
    
    dataset_generators = {
        "kubric_sync": generate_kubric_sync,
        "kubric_spatial": generate_kubric_spatial,
        "clevrer_temporal": generate_clevrer_temporal,
        "clevrer_comparative": generate_clevrer_comparative,
        "habitat_tracking": generate_habitat_tracking,
        "habitat_holistic": generate_habitat_holistic,
    }
    
    generator = dataset_generators[args.dataset]
    
    try:
        generator(args)
        print("\n" + "="*70)
        print("Data generation completed successfully!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nError during data generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

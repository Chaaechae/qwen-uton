# Copyright (c) 2026
# Preprocessing script to convert Video-3D-LLM dataset
# (https://huggingface.co/datasets/zd11024/Video-3D-LLM_data)
# into Utonia-compatible input format for semantic segmentation.

import os
import argparse
import glob
import numpy as np
import torch


def load_video3dllm_pth(pth_path: str) -> dict:
    """Load a single scene .pth file from Video-3D-LLM pcd_with_object_aabbs/.

    Expected keys in .pth:
        xyz: (N, 3) float32 - vertex positions
        rgb: (N, 3) uint8   - vertex colors [0-255]
        normal: (N, 3) float32 - vertex normals
        sem_labels: (N,) int  - semantic labels (-1 = unlabeled, 0-19 = ScanNet20)
        instance_ids: (N,) int - instance IDs
        aabb_obj_ids: (M,) int - bounding box object IDs
        aabb_corner_xyz: (M, 8, 3) float32 - AABB corners
    """
    data = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Handle both tensor and numpy values
    result = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.numpy()
        else:
            result[key] = np.asarray(val)
    return result


def convert_to_utonia(data: dict) -> dict:
    """Convert Video-3D-LLM data dict to Utonia input format.

    Utonia expects:
        coord:   (N, 3) float64 - XYZ coordinates
        color:   (N, 3) float64 - RGB [0-255]
        normal:  (N, 3) float64 - surface normals
        segment: (N,)   int     - semantic labels (optional)
    """
    coord = data["xyz"].astype(np.float64)
    color = data["rgb"].astype(np.float64)

    if "normal" in data:
        normal = data["normal"].astype(np.float64)
    else:
        normal = np.zeros_like(coord)

    point = {
        "coord": coord,
        "color": color,
        "normal": normal,
    }

    # Map Video-3D-LLM sem_labels to Utonia segment format.
    # Video-3D-LLM uses ScanNet labels: -1 = unlabeled, 0-19 = class indices.
    # Utonia sample1 uses the same ScanNet20 label scheme directly.
    if "sem_labels" in data:
        sem_labels = data["sem_labels"].astype(np.int64)
        point["segment"] = sem_labels

    return point


def save_as_npz(point: dict, output_path: str):
    """Save Utonia-compatible point dict as .npz file."""
    np.savez_compressed(output_path, **point)
    print(f"Saved: {output_path} ({point['coord'].shape[0]} points)")


def convert_single(pth_path: str, output_path: str):
    """Convert a single .pth file to Utonia .npz format."""
    data = load_video3dllm_pth(pth_path)
    point = convert_to_utonia(data)
    save_as_npz(point, output_path)
    return point


def convert_directory(input_dir: str, output_dir: str):
    """Batch convert all .pth files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    pth_files = sorted(glob.glob(os.path.join(input_dir, "*.pth")))
    if not pth_files:
        print(f"No .pth files found in {input_dir}")
        return

    print(f"Found {len(pth_files)} scenes to convert")
    for pth_path in pth_files:
        scene_id = os.path.splitext(os.path.basename(pth_path))[0]
        output_path = os.path.join(output_dir, f"{scene_id}.npz")
        convert_single(pth_path, output_path)

    print(f"Done. Converted {len(pth_files)} scenes to {output_dir}")


def run_semantic_segmentation(npz_path: str, save_ply: str = None):
    """Run Utonia semantic segmentation on a converted .npz file.

    This is a self-contained demo that loads the converted data,
    runs the model, and visualizes/saves the result.
    """
    import utonia
    import torch.nn as nn

    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    try:
        import flash_attn
    except ImportError:
        flash_attn = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ScanNet 20-class labels and colors (same as demo/2_sem_seg.py)
    VALID_CLASS_IDS_20 = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
    )
    CLASS_LABELS_20 = (
        "wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "bookshelf",
        "picture", "counter", "desk", "curtain", "refrigerator",
        "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
    )
    SCANNET_COLOR_MAP_20 = {
        0: (0., 0., 0.), 1: (174., 199., 232.), 2: (152., 223., 138.),
        3: (31., 119., 180.), 4: (255., 187., 120.), 5: (188., 189., 34.),
        6: (140., 86., 75.), 7: (255., 152., 150.), 8: (214., 39., 40.),
        9: (197., 176., 213.), 10: (148., 103., 189.), 11: (196., 156., 148.),
        12: (23., 190., 207.), 14: (247., 182., 210.), 15: (66., 188., 102.),
        16: (219., 219., 141.), 17: (140., 57., 197.), 18: (202., 185., 52.),
        19: (51., 176., 203.), 20: (200., 54., 131.),
    }
    CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]

    class SegHead(nn.Module):
        def __init__(self, backbone_out_channels, num_classes):
            super().__init__()
            self.seg_head = nn.Linear(backbone_out_channels, num_classes)

        def forward(self, x):
            return self.seg_head(x)

    # Load model
    print("Loading Utonia model...")
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = utonia.load(
            "utonia", repo_id="Pointcept/Utonia", custom_config=custom_config
        ).to(device)

    # Load seg head
    ckpt = utonia.load(
        "utonia_linear_prob_head_sc",
        repo_id="Pointcept/Utonia",
        ckpt_only=True,
    )
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])

    # Load converted data
    print(f"Loading point cloud: {npz_path}")
    point = dict(np.load(npz_path))

    # Remove segment for inference (keep a copy for evaluation if present)
    gt_segment = point.pop("segment", None)

    # Apply default transform
    transform = utonia.transform.default(0.5)
    point = transform(point)

    # Inference
    print("Running inference...")
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)

        # Upcast features through all pooling levels
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        seg_logits = seg_head(point.feat)
        pred = seg_logits.argmax(dim=-1).cpu().numpy()
        pred_color = np.array(CLASS_COLOR_20)[pred]

    n_points = point.coord.shape[0]
    print(f"Segmentation complete: {n_points} points")

    # Print class distribution
    unique, counts = np.unique(pred, return_counts=True)
    print("\nPredicted class distribution:")
    for cls_idx, cnt in zip(unique, counts):
        pct = 100.0 * cnt / n_points
        print(f"  {CLASS_LABELS_20[cls_idx]:20s}: {cnt:8d} ({pct:5.1f}%)")

    # Evaluate against GT if available
    if gt_segment is not None:
        valid_mask = gt_segment >= 0  # -1 = unlabeled
        if valid_mask.any():
            # Note: GT may be at original resolution, pred at grid-sampled resolution
            print(f"\nGT labels available ({valid_mask.sum()} labeled points in original)")

    # Visualize / save
    if save_ply:
        if o3d is None:
            print("open3d not installed, skipping PLY save")
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                point.coord.cpu().detach().numpy()
            )
            pcd.colors = o3d.utility.Vector3dVector(pred_color / 255.0)
            o3d.io.write_point_cloud(save_ply, pcd)
            print(f"Saved visualization: {save_ply}")
    else:
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                point.coord.cpu().detach().numpy()
            )
            pcd.colors = o3d.utility.Vector3dVector(pred_color / 255.0)
            o3d.visualization.draw_geometries([pcd])
        else:
            print("open3d not installed, skipping visualization")

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Video-3D-LLM data to Utonia format and run semantic segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- convert command ---
    p_convert = subparsers.add_parser(
        "convert", help="Convert .pth files to Utonia .npz format"
    )
    p_convert.add_argument(
        "input", help="Path to a single .pth file or directory of .pth files"
    )
    p_convert.add_argument(
        "-o", "--output",
        help="Output path (.npz file or directory). "
             "Defaults to same location with .npz extension.",
    )

    # --- run command ---
    p_run = subparsers.add_parser(
        "run", help="Run semantic segmentation on a converted .npz file"
    )
    p_run.add_argument("input", help="Path to a .npz file (converted)")
    p_run.add_argument(
        "--save-ply", default=None,
        help="Save result as PLY file instead of displaying",
    )

    # --- convert-and-run command ---
    p_both = subparsers.add_parser(
        "convert-and-run",
        help="Convert a .pth file and immediately run segmentation",
    )
    p_both.add_argument("input", help="Path to a .pth file")
    p_both.add_argument(
        "--save-ply", default=None,
        help="Save result as PLY file instead of displaying",
    )

    args = parser.parse_args()

    if args.command == "convert":
        if os.path.isfile(args.input):
            output = args.output or args.input.replace(".pth", ".npz")
            convert_single(args.input, output)
        elif os.path.isdir(args.input):
            output = args.output or os.path.join(args.input, "utonia_npz")
            convert_directory(args.input, output)
        else:
            raise FileNotFoundError(f"Input not found: {args.input}")

    elif args.command == "run":
        run_semantic_segmentation(args.input, save_ply=args.save_ply)

    elif args.command == "convert-and-run":
        # Convert to temp npz, then run
        tmp_npz = args.input.replace(".pth", "_utonia.npz")
        convert_single(args.input, tmp_npz)
        run_semantic_segmentation(tmp_npz, save_ply=args.save_ply)

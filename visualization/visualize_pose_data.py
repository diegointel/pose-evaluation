import os
import numpy as np
import pandas as pd
import torch
import visdom
import argparse
from easydict import EasyDict as edict

import matplotlib
matplotlib.use('Agg')  # if running in a headless environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import camera  # Your camera transformations module, required by get_camera_mesh

def get_camera_mesh(pose_4x4, depth=1):
    # shape: pose_4x4 = [N,4,4]
    # reduce to [N,3,4] by removing the last row
    pose_3x4 = pose_4x4[..., :3, :]  # keep only rows 0..2

    # build local camera frustum (5 points)
    vertices = torch.tensor([
        [-0.5, -0.5, 1],
        [0.5, -0.5, 1],
        [0.5,  0.5, 1],
        [-0.5,  0.5, 1],
        [0.0,  0.0, 0],
    ]) * depth

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ])

    vertices = camera.cam2world(vertices[None], pose_3x4)
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]+[None]
    return wireframe_merged


def merge_meshes(vertices, faces):
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces + i*vertex_N for i in range(mesh_N)], dim=0)
    vertices_merged = vertices.view(-1, vertices.shape[-1])
    return vertices_merged, faces_merged


def merge_centers(two_centers):
    """Just merges the centers of two sets of cameras for line-plotting."""
    c1, c2 = two_centers
    center_merged = [[],[],[]]
    for cc1, cc2 in zip(c1, c2):
        center_merged[0] += [float(cc1[0]), float(cc2[0]), None]
        center_merged[1] += [float(cc1[1]), float(cc2[1]), None]
        center_merged[2] += [float(cc1[2]), float(cc2[2]), None]
    return center_merged


@torch.no_grad()
def vis_cameras(opt, vis, step, poses=[], colors=["blue", "magenta"], plot_dist=True):
    """
    Visualize multiple sets of poses in 3D using Visdom.
    Each element in 'poses' is a PyTorch tensor of shape [N, 4, 4].
    """
    win_name = "{}/{}".format(opt.group, opt.name)
    data = []
    centers = []
    for pose, color in zip(poses, colors):
        pose = pose.detach().cpu()
        vertices, faces, wireframe = get_camera_mesh(pose, depth=opt.visdom.cam_depth)
        center = vertices[:, -1]
        centers.append(center)
        # Camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:, 0]],
            y=[float(n) for n in center[:, 1]],
            z=[float(n) for n in center[:, 2]],
            mode="markers",
            marker=dict(color=color, size=3),
        ))
        # Camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:, 0]],
            y=[float(n) for n in vertices_merged[:, 1]],
            z=[float(n) for n in vertices_merged[:, 2]],
            i=[int(n) for n in faces_merged[:, 0]],
            j=[int(n) for n in faces_merged[:, 1]],
            k=[int(n) for n in faces_merged[:, 2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # Camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color),
            opacity=0.3,
        ))
    # Optionally plot distance lines between pose 0 and 1, etc.
    if plot_dist and len(centers) >= 2:
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red", width=4),
        ))
    # Send data to Visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title=f"({step})",
            autosize=True,
            margin=dict(l=30, r=30, b=30, t=30),
            showlegend=False,
        ),
        opts=dict(title=f"{win_name} poses ({step})")
    ))


def parse_pose_string(pose_str):
    """
    Parse a string representation of a flattened 4x4 matrix.
    Example of 'pose_str': "[0. 1. 2. 3. 4. 5. ... 0. ]"
    """
    # Remove brackets if present
    pose_str = pose_str.strip("[]")
    # Convert to float array
    vals = np.fromstring(pose_str, sep=' ')
    if len(vals) != 16:
        raise ValueError(f"Expecting 16 values for a 4x4 matrix, but got {len(vals)}.")
    return vals.reshape(4,4)

def load_comparison_csv(csv_path):
    """
    Load the CSV file containing columns:
        'Image ID', 'Ground Truth Pose', 'Other Pose', 'Difference'
    Return two lists of 4x4 numpy arrays: (gt_poses, other_poses).
    """
    df = pd.read_csv(csv_path)
    gt_poses = []
    other_poses = []
    for _, row in df.iterrows():
        gt_pose = parse_pose_string(str(row['Ground Truth Pose']))
        other_pose = parse_pose_string(str(row['Other Pose']))
        gt_poses.append(gt_pose)
        other_poses.append(other_pose)
    # Convert to torch tensors for visualization
    gt_poses = torch.from_numpy(np.stack(gt_poses, axis=0)).float()
    other_poses = torch.from_numpy(np.stack(other_poses, axis=0)).float()
    return gt_poses, other_poses

def visualize_folder(folder, vis, opt):
    """
    For a given folder, load the three CSV files (if present):
        - colmap_comparison.csv
        - colmap_free3dgs_comparison.csv
        - gaussian_object_comparison.csv
    and visualize them in Visdom.
    """
    output_folder = f"pose_data_comparation/{folder}"
    # Example file names
    csv_files = ["colmap_comparison.csv", "colmap_free3dgs_comparison.csv", "gaussian_object_comparison.csv", "colmap_.csv", "colmap_free3dgs_.csv", "gaussian_object_.csv", "ground_truth_.csv"]
    method_names = ["Colmap", "ColmapFree3DGS", "GaussianObject", "Colmap_raw", "ColmapFree3DGS_raw", "GaussianObject_raw", "GroundTruth_raw"]

    for csv_file, method_name in zip(csv_files, method_names):
        csv_path = os.path.join(output_folder, csv_file)
        if not os.path.exists(csv_path):
            print(f"[WARNING] File {csv_path} does not exist. Skipping.")
            continue

        # Load data
        gt_poses, method_poses = load_comparison_csv(csv_path)
        # Visualize
        opt.group = folder
        opt.name = method_name
        print(f"Visualizing {method_name} for folder {folder} with {len(gt_poses)} pairs of poses...")
        vis_cameras(opt, vis, step=0, poses=[gt_poses, method_poses],
                    colors=["green", "magenta"], plot_dist=True)


def visualize_test_folder(folder, vis, opt):
    output_folder = f"pose_data_comparation/{folder}"
    csv_files = ["scaled_comparison.csv", "rotated_comparison.csv", "rotated_scaled_comparison.csv", "scaled_.csv", "rotated_.csv", "rotated_scaled_.csv", "ground_truth_.csv"]
    method_names = ["scaled", "rotated", "rotated_scaled", "scaled_raw", "rotated_raw", "rotated_scaled_raw", "GroundTruth_raw"]

    for csv_file, method_name in zip(csv_files, method_names):
        csv_path = os.path.join(output_folder, csv_file)
        if not os.path.exists(csv_path):
            print(f"[WARNING] File {csv_path} does not exist. Skipping.")
            continue

        # Load data
        gt_poses, method_poses = load_comparison_csv(csv_path)
        # Visualize
        opt.group = folder
        opt.name = method_name
        print(f"Visualizing {method_name} for folder {folder} with {len(gt_poses)} pairs of poses...")
        vis_cameras(opt, vis, step=0, poses=[gt_poses, method_poses],
                    colors=["green", "magenta"], plot_dist=True)


def normalize_camera_poses_with_steps(camera_poses):
    """
    Normalize camera poses with intermediate steps for visualization.

    Args:
        camera_poses (dict): Dictionary of camera poses.

    Returns:
        dict: Final normalized poses.
        list: Intermediate steps for visualization.
    """
    if not camera_poses:
        return {}, []

    # Select reference camera (camera 0)
    reference_image_id = next(iter(camera_poses))
    reference_pose = camera_poses[reference_image_id]

    # Extract rotation (R) and translation (t) of reference camera
    R_ref = reference_pose[:3, :3]
    t_ref = reference_pose[:3, 3]

    # Ensure the reference camera is at the origin with identity rotation
    T_ref = np.eye(4)

    # Step 1: Original poses
    intermediate_steps = [{"step": "Original", "poses": camera_poses.copy()}]

    # Step 2: Move reference camera to the origin
    moved_poses = {}
    for image_id, pose in camera_poses.items():
        R_current = pose[:3, :3]
        t_current = pose[:3, 3]

        R_relative = np.dot(np.linalg.inv(R_ref), R_current)
        t_relative = t_current - t_ref

        T_normalized = np.eye(4)
        T_normalized[:3, :3] = R_relative
        T_normalized[:3, 3] = t_relative
        moved_poses[image_id] = T_normalized

    intermediate_steps.append({"step": "Moved to Origin", "poses": moved_poses})

    # Step 3: Scale distances
    distances = []
    for image_id, pose in moved_poses.items():
        if image_id != reference_image_id:
            t_current = pose[:3, 3]
            distances.append(np.linalg.norm(t_current))
    avg_distance = np.mean(distances) if distances else 1.0
    scale_factor = 1.0 / avg_distance

    scaled_poses = {}
    for image_id, pose in moved_poses.items():
        pose[:3, 3] *= scale_factor
        scaled_poses[image_id] = pose

    intermediate_steps.append({"step": "Scaled Distances", "poses": scaled_poses})

    # Final step: Normalized poses
    intermediate_steps.append({"step": "Final Normalized", "poses": scaled_poses})
    return scaled_poses, intermediate_steps


def visualize_normalization_folder(folder, vis, opt):
    output_folder = f"pose_data_comparation/{folder}"
    csv_path = os.path.join(output_folder, "ground_truth_.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] File {csv_path} does not exist. Skipping.")
        return

    # Load ground truth poses
    gt_poses, _ = load_comparison_csv(csv_path)
    gt_poses_dict = {f"Camera {i}": pose.numpy() for i, pose in enumerate(gt_poses)}

    # Normalize poses and get intermediate steps
    normalized_poses, steps = normalize_camera_poses_with_steps(gt_poses_dict)

    # Visualize each step
    for step_info in steps:
        step_name = step_info["step"]
        step_poses = torch.tensor(np.stack(list(step_info["poses"].values())))
        opt.name = f"{folder}_{step_name}"
        print(f"Visualizing {step_name} for folder {folder}...")
        vis_cameras(opt, vis, step=0, poses=[step_poses], colors=["blue"], plot_dist=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", default=["AP13","MDF14","ShSu12","SMu1"],
                        help="List of folders to visualize (default: AP13 MDF14 ShSu12 SMu1)")
    parser.add_argument("--visdom_server", type=str, default="http://localhost",
                        help="Visdom server URL (default: http://localhost)")
    parser.add_argument("--visdom_port", type=int, default=8097,
                        help="Visdom server port (default: 8097)")
    parser.add_argument("--cam_depth", type=float, default=1.0,
                        help="Size/depth scale of the camera frustum in the plot.")
    args = parser.parse_args()

    # Setup Visdom
    vis = visdom.Visdom(server=args.visdom_server, port=args.visdom_port)
    if not vis.check_connection():
        print("Could not connect to Visdom server. Make sure it is running:")
        print("    python -m visdom.server")
        return

    # Prepare an "opt" object for the vis_cameras function
    opt = edict()
    opt.visdom = edict()
    opt.visdom.cam_depth = args.cam_depth
    # 'group' and 'name' will be set dynamically inside visualize_folder

    for folder in args.folders:
        visualize_folder(folder, vis, opt)
    visualize_test_folder("test_rotation", vis, opt)
    visualize_normalization_folder("MDF14", vis, opt)

if __name__ == "__main__":
    main()

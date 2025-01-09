import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_camera_poses(file_path, id_column, pose_column, image_id_format, multi_column=False):
    """
    Load camera poses from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        id_column (str): Column name for image IDs.
        pose_column (str): Column name for camera poses.
        image_id_format (callable): Function to normalize image ID format.
        multi_column (bool): Whether pose data spans multiple columns.

    Returns:
        dict: Dictionary mapping image IDs to camera poses.
    """
    camera_poses = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = image_id_format(row[id_column])
            if multi_column:
                pose_values = []
                for key, value in row.items():
                    if key != id_column and value:
                        if isinstance(value, list):
                            # Flatten the list and append each value
                            pose_values.extend(float(v) for v in value)
                        elif isinstance(value, (str, int, float)):
                            # Convert individual values to float
                            pose_values.append(float(value))
                        else:
                            raise TypeError(f"Unexpected data type: {type(value)} for key {key}")
                pose = np.array(pose_values)
            else:
                pose_str = row[pose_column].replace('[', '').replace(']', '').replace('"', '').replace(',', '')
                pose = np.fromstring(pose_str, sep=' ')
            camera_poses[image_id] = pose.reshape(4, 4) if len(pose) == 16 else convert_to_4x4(pose)
    return camera_poses



def convert_to_4x4(pose):
    """
    Convert a flat pose array into a 4x4 transformation matrix.

    Args:
        pose (np.ndarray): Flat pose array.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    normalized_pose = np.eye(4)
    normalized_pose[:3, :3] = pose[:9].reshape(3, 3)
    normalized_pose[:3, 3] = pose[9:12]
    return normalized_pose


# def normalize_camera_poses(camera_poses):
#     """
#     Normalize camera poses: move camera 0 to the origin, scale distances,
#     and align directions for consistency.

#     Args:
#         camera_poses (dict): Dictionary of camera poses.

#     Returns:
#         dict: Dictionary of normalized camera poses.
#     """
#     if not camera_poses:
#         return {}

#     # Select reference camera (camera 0)
#     reference_image_id = next(iter(camera_poses))
#     reference_pose = camera_poses[reference_image_id]

#     # Extract rotation and translation of camera 0
#     R_ref = reference_pose[:3, :3]
#     t_ref = reference_pose[:3, 3]

#     # Force reference rotation to identity
#     T_ref = np.eye(4)
#     T_ref[:3, :3] = np.eye(3)
#     T_ref[:3, 3] = [0, 0, 0]

#     # Identify a second camera to calculate scale
#     second_image_id = None
#     for image_id in camera_poses:
#         if image_id != reference_image_id:
#             second_image_id = image_id
#             break

#     if not second_image_id:
#         return {reference_image_id: T_ref}

#     # Calculate scale based on distance between camera 0 and camera 1
#     ref_translation = reference_pose[:3, 3]
#     second_translation = camera_poses[second_image_id][:3, 3]
#     original_distance = np.linalg.norm(second_translation - ref_translation)

#     scale_factor = 1.0 / (original_distance if original_distance != 0 else 1)

#     # Normalize all cameras
#     normalized_poses = {}
#     for image_id, pose in camera_poses.items():
#         # R_relative = np.dot(np.linalg.inv(R_ref), pose[:3, :3])  # Relative rotation to camera 0
#         R_relative = pose[:3, :3]
#         t_relative = scale_factor * (pose[:3, 3] - t_ref)        # Relative translation (scaled)

#         T_normalized = np.eye(4)
#         T_normalized[:3, :3] = R_relative
#         T_normalized[:3, 3] = t_relative
#         normalized_poses[image_id] = T_normalized

#     # Ensure camera 0 is at the origin with identity rotation
#     # normalized_poses[reference_image_id] = T_ref

#     return normalized_poses


def normalize_camera_poses(camera_poses):
    """
    Normalize camera poses: move camera 0 to the origin, scale distances,
    and align directions for consistency.

    Args:
        camera_poses (dict): Dictionary of camera poses.

    Returns:
        dict: Dictionary of normalized camera poses.
    """
    if not camera_poses:
        return {}

    # Select reference camera (camera 0)
    reference_image_id = next(iter(camera_poses))
    reference_pose = camera_poses[reference_image_id]

    # Extract rotation (R) and translation (t) of reference camera
    R_ref = reference_pose[:3, :3]
    t_ref = reference_pose[:3, 3]

    # Ensure the reference camera is at the origin with identity rotation
    T_ref = np.eye(4)

    # Calculate scale using all cameras
    distances = []
    ref_translation = t_ref
    for image_id, pose in camera_poses.items():
        if image_id != reference_image_id:
            second_translation = pose[:3, 3]
            distance = np.linalg.norm(second_translation - ref_translation)
            distances.append(distance)
    avg_distance = np.mean(distances) if distances else 1.0
    scale_factor = 1.0 / avg_distance

    # Normalize all cameras
    normalized_poses = {}
    for image_id, pose in camera_poses.items():
        R_current = pose[:3, :3]
        t_current = pose[:3, 3]

        # Compute relative rotation and translation
        R_relative = np.dot(np.linalg.inv(R_ref), R_current)
        t_relative = scale_factor * (t_current - t_ref)

        # Construct normalized pose
        T_normalized = np.eye(4)
        T_normalized[:3, :3] = R_relative
        T_normalized[:3, 3] = t_relative
        normalized_poses[image_id] = T_normalized

    # Explicitly set the reference camera to identity pose
    normalized_poses[reference_image_id] = T_ref

    return normalized_poses



def filter_and_normalize_poses(method_poses, ground_truth_poses):
    """
    Filter images used by the method, pair with GroundTruth, and normalize.

    Args:
        method_poses (dict): Poses from a specific method.
        ground_truth_poses (dict): GroundTruth poses.

    Returns:
        dict: Normalized poses from the method.
        dict: Corresponding normalized GroundTruth poses.
    """
    common_ids = set(method_poses.keys()) & set(ground_truth_poses.keys())
    method_filtered = {image_id: method_poses[image_id] for image_id in common_ids}
    ground_truth_filtered = {image_id: ground_truth_poses[image_id] for image_id in common_ids}
    method_normalized = normalize_camera_poses(method_filtered)
    ground_truth_normalized = normalize_camera_poses(ground_truth_filtered)
    return method_normalized, ground_truth_normalized


def extract_centers_from_poses(camera_poses):
    """
    From a dict {id -> 4x4 pose}, return:
    - centers: Nx3 array of translation centers,
    - ids: list of pose IDs in the same order,
    - poses_4x4: list of the original 4x4 matrices.
    """
    ids = list(camera_poses.keys())
    centers = []
    poses_4x4 = []
    for img_id in ids:
        pose = camera_poses[img_id]
        center = pose[:3, 3]  # translation is the last column of the top 3 rows
        centers.append(center)
        poses_4x4.append(pose)
    centers = np.vstack(centers)  # shape (N,3)
    return centers, ids, poses_4x4

def procrustes_align_3D(src_points, tgt_points, do_scale=True):
    """
    Align src_points -> tgt_points in 3D using an SVD-based Procrustes method.
    If do_scale=False, we preserve the original scale (s=1).

    Args:
        src_points (ndarray): (N,3) source points
        tgt_points (ndarray): (N,3) target points
        do_scale (bool): If True, find a global scale s. If False, keep s=1.

    Returns:
        R (3x3), s (float), t (3,): rotation, scale, translation
    """
    # 1) Compute the centroids
    src_centroid = src_points.mean(axis=0)
    tgt_centroid = tgt_points.mean(axis=0)
    src_centered = src_points - src_centroid
    tgt_centered = tgt_points - tgt_centroid

    # 2) If do_scale=True, normalize each set by its RMS distance
    if do_scale:
        src_norm = np.sqrt((src_centered**2).sum())
        tgt_norm = np.sqrt((tgt_centered**2).sum())
        # Avoid division by zero
        if src_norm < 1e-9 or tgt_norm < 1e-9:
            # fallback to no scale
            src_norm, tgt_norm = 1.0, 1.0
        src_centered /= src_norm
        tgt_centered /= tgt_norm

    # 3) Compute rotation via SVD
    H = src_centered.T @ tgt_centered  # (3xN)(N×3) = 3×3
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Fix reflection if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 4) Compute scale
    if do_scale:
        # ratio of the sums of squares
        src_ssq = (src_centered**2).sum()
        tgt_ssq = (tgt_centered**2).sum()
        s = np.sqrt(tgt_ssq / (src_ssq + 1e-9))
    else:
        s = 1.0

    # 5) Compute translation
    #   aligned_src = s*R*(src - src_centroid) + t = tgt
    # => t = tgt_centroid - s*R*src_centroid
    t = tgt_centroid - s * (R @ src_centroid)

    return R, s, t


def apply_rigid_transform_4x4(pose_4x4, R, s, t):
    """
    Apply the transformation (R, s, t) to a 4x4 camera pose matrix.
    pose_4x4 = [R0 | t0]
               [0 0 0 1 ]
    """
    # Decompose the original pose
    R0 = pose_4x4[:3, :3]
    t0 = pose_4x4[:3, 3]

    # Compose new rotation, translation
    R_new = R @ R0
    t_new = s * (R @ t0) + t

    # Build a new 4x4
    pose_new = np.eye(4)
    pose_new[:3, :3] = R_new
    pose_new[:3, 3] = t_new
    return pose_new

def align_camera_poses_procrustes(ref_poses, tgt_poses, do_scale=True):
    """
    Align tgt_poses to ref_poses (both dicts: {id -> 4x4}).
    Returns aligned_tgt_poses (dict).
    """
    # 1) Find the common IDs (to match)
    common_ids = set(ref_poses.keys()).intersection(set(tgt_poses.keys()))
    if len(common_ids) < 2:
        print("[Warning] <2 common IDs, can't do Procrustes. Returning original.")
        return dict(tgt_poses)  # no alignment

    # 2) Extract centers for these common IDs, build arrays
    ref_centers, ref_order = [], []
    tgt_centers, tgt_order = [], []
    for cid in sorted(common_ids):
        ref_centers.append(ref_poses[cid][:3, 3])
        tgt_centers.append(tgt_poses[cid][:3, 3])
        ref_order.append(cid)
        tgt_order.append(cid)
    ref_centers = np.vstack(ref_centers)  # (M,3)
    tgt_centers = np.vstack(tgt_centers)  # (M,3)

    # 3) Compute transform that aligns tgt->ref
    R, s, t = procrustes_align_3D(tgt_centers, ref_centers, do_scale=do_scale)

    # 4) Apply (R,s,t) to *all* tgt_poses (not just the common ones)
    aligned = {}
    for cid, pose_4x4 in tgt_poses.items():
        aligned[cid] = apply_rigid_transform_4x4(pose_4x4, R, s, t)

    return aligned

def filter_and_align_poses(method_poses, ground_truth_poses, do_scale=True):
    """
    Filter out only the common IDs, then align method_poses to ground_truth_poses
    via Procrustes (R, s, t). Return (aligned_method_poses, filtered_ground_truth).

    Args:
        method_poses (dict): Dictionary of method poses.
        ground_truth_poses (dict): Dictionary of ground truth poses.
        do_scale (bool): Whether to apply scaling during alignment.

    Returns:
        dict: Aligned poses from the method.
        dict: Filtered ground truth poses.
    """
    # Find common IDs
    common_ids = set(method_poses.keys()) & set(ground_truth_poses.keys())
    if not common_ids:
        return {}, {}

    # Filter each dictionary
    method_filtered = {img_id: method_poses[img_id] for img_id in common_ids}
    gt_filtered = {img_id: ground_truth_poses[img_id] for img_id in common_ids}

    # Normalize both sets of poses
    method_normalized = normalize_camera_poses(method_filtered)
    gt_normalized = normalize_camera_poses(gt_filtered)

    # Align the normalized poses
    aligned_method_poses = align_camera_poses_procrustes(
        gt_normalized, method_normalized, do_scale=do_scale
    )

    return aligned_method_poses, gt_normalized




def compare_camera_poses(ground_truth_poses, other_poses):
    """
    Compare camera poses between GroundTruth and another method.

    Args:
        ground_truth_poses (dict): GroundTruth poses.
        other_poses (dict): Poses from another method.

    Returns:
        pd.DataFrame: Comparison results.
    """
    comparison_results = []
    for image_id, gt_pose in ground_truth_poses.items():
        if image_id in other_poses:
            other_pose = other_poses[image_id]
            difference = np.linalg.norm(gt_pose - other_pose)
            comparison_results.append({
                'Image ID': image_id,
                'Ground Truth Pose': gt_pose.flatten(),
                'Other Pose': other_pose.flatten(),
                'Difference': difference
            })
    return pd.DataFrame(comparison_results)


def normalize_image_id(image_id):
    """
    Normalize image ID format.

    Args:
        image_id (str): Original image ID.

    Returns:
        str: Normalized image ID.
    """
    return image_id.zfill(4) if image_id.isdigit() else os.path.splitext(image_id)[0]


def main():
    data_folders = ['AP13', 'MDF14', 'ShSu12', 'SMu1']
    # data_folders = ['test_rotation']
    for folder in data_folders:
        ground_truth_path = f'data/poses_output/{folder}/ground_truth_camera_poses.csv'
        colmap_path = f'data/ALGO_poses_result/COLMAP/{folder}/camera_positions.csv'
        colmap_free3dgs_path = f'data/ALGO_poses_result/CF3DGS/HO3Dv3_{folder}/poses_pred.csv'
        gaussian_object_path = f'data/ALGO_poses_result/GAUSSIAN_OBJECT/{folder}/camera_poses.csv'
        outputFolder = f'pose_data_comparation/{folder}/'
        os.makedirs(outputFolder, exist_ok=True)

        ground_truth_poses = load_camera_poses(ground_truth_path, 'Image ID', 'Camera Pose', normalize_image_id)
        colmap_poses = load_camera_poses(colmap_path, 'Image Name', 'Camera Pose', normalize_image_id)
        colmap_free3dgs_poses = load_camera_poses(colmap_free3dgs_path, 'Image ID', 'Camera Pose', normalize_image_id, multi_column=True)
        gaussian_object_poses = load_camera_poses(gaussian_object_path, 'Image Name', 'Camera Pose', normalize_image_id)

        max_len_set = len(colmap_poses)
        max_set = colmap_poses
        if (len(colmap_free3dgs_poses) > max_len_set):
            max_len_set = len(colmap_free3dgs_poses)
            max_set = colmap_free3dgs_poses
        elif (len(gaussian_object_poses) > max_len_set):
            max_len_set = len(gaussian_object_poses)
            max_set = gaussian_object_poses

        common_ids = set(max_set.keys()) & set(ground_truth_poses.keys())
        if not common_ids:
            return {}, {}

        # Filter each dictionary
        gt_filtered = {img_id: ground_truth_poses[img_id] for img_id in common_ids}
        
        colmap_ = compare_camera_poses(colmap_poses, colmap_poses)
        colmap_free3dgs_ = compare_camera_poses(colmap_free3dgs_poses, colmap_free3dgs_poses)
        gaussian_object_ = compare_camera_poses(gaussian_object_poses, gaussian_object_poses)
        ground_truth_ = compare_camera_poses(gt_filtered, gt_filtered)

        colmap_poses, colmap_ground_truth = filter_and_align_poses(colmap_poses, ground_truth_poses, do_scale=True)
        colmap_free3dgs_poses, colmap_free3dgs_ground_truth = filter_and_align_poses(colmap_free3dgs_poses, ground_truth_poses, do_scale=True)
        gaussian_object_poses, gaussian_object_ground_truth = filter_and_align_poses(gaussian_object_poses, ground_truth_poses, do_scale=True)

        colmap_comparison = compare_camera_poses(colmap_ground_truth, colmap_poses)
        colmap_free3dgs_comparison = compare_camera_poses(colmap_free3dgs_ground_truth, colmap_free3dgs_poses)
        gaussian_object_comparison = compare_camera_poses(gaussian_object_ground_truth, gaussian_object_poses)

    
        colmap_comparison.to_csv(f'{outputFolder}/colmap_comparison.csv', index=False)
        colmap_free3dgs_comparison.to_csv(f'{outputFolder}/colmap_free3dgs_comparison.csv', index=False)
        gaussian_object_comparison.to_csv(f'{outputFolder}/gaussian_object_comparison.csv', index=False)
        colmap_.to_csv(f'{outputFolder}/colmap_.csv', index=False)
        colmap_free3dgs_.to_csv(f'{outputFolder}/colmap_free3dgs_.csv', index=False)
        gaussian_object_.to_csv(f'{outputFolder}/gaussian_object_.csv', index=False)
        ground_truth_.to_csv(f'{outputFolder}/ground_truth_.csv', index=False)

if __name__ == "__main__":
    main()

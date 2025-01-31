import os
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def compute_rpe(view_matrices, ref_gt, get_raw_errors=False):
    trans_errors = []
    rot_errors = []

    list_vm = list(view_matrices.values())
    list_gt = list(ref_gt.values())

    for i in range(len(list_gt) - 1):
        gt1 = list_gt[i]
        gt2 = list_gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = list_vm[i]
        pred2 = list_vm[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))

    if get_raw_errors:
        return trans_errors, rot_errors

    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


def compute_ATE(view_matrices, ref_gt):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []
    for key in view_matrices.keys():
        _, pred_xyz = get_Rt_2_world(view_matrices[key])
        _, gt_xyz = get_Rt_2_world(ref_gt[key])

        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err**2)))

    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
    return ate


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

def get_Rt_2_world(view_matrix):
    R = view_matrix[:3, :3]
    t_c = view_matrix[:3, 3]
    t = -R.T @ t_c
    return R, t

def take_Rts_2_world(view_matrices):
    R_set = {}
    t_set = {}

    # Take Rtp
    for (key, val) in view_matrices.items():
        R, t = get_Rt_2_world(val)
        R_set[key] = R
        t_set[key] = t

    return R_set, t_set

def move_centroid_to_origin(view_matrices):

    # Take positions
    _, t_set = take_Rts_2_world(view_matrices)

    # Calculate centroid
    average_vector = np.average(list(t_set.values()), axis=0)

    # Move to origin
    for val in t_set.values():
        val -= average_vector

    # Save new translations
    for key in view_matrices.keys():
        view_matrices[key][:3, 3] = -view_matrices[key][:3, :3] @ t_set[key]

def normalize_p_to_unit_sphere(view_matrices):
    s = 1

    # Take positions
    R_set, t_set = take_Rts_2_world(view_matrices)

    # Compute norms of camera positions
    norms = [np.linalg.norm(pos) for pos in t_set.values()]

    # Find maximum norm
    max_norm = max(norms)

    if (max_norm < 0.98 * s) or (max_norm > 1.01*s):
        for key in view_matrices.keys():
            new_p = (t_set[key]/max_norm)*s
            view_matrices[key][:3, 3] = (-R_set[key] @ new_p)

def normalize_rotation(view_matrices):
    for key in view_matrices.keys():
        # Extract rotation matrix (upper-left 3x3 block)
        R, _ = get_Rt_2_world(view_matrices[key])

        # Normalize the rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R)
        R_normalized = np.dot(U, Vt)

        # Ensure determinant is +1 (if det(R_normalized) = -1, correct it)
        if np.linalg.det(R_normalized) < 0:
            R_normalized *= -1

        # Reassemble the normalized rotation matrix
        view_matrices[key][:3, :3] = R_normalized

def scale_method_to_gt(view_matrices, ref_gt):

    # Take positions
    R_set, t_set_vm = take_Rts_2_world(view_matrices)
    _, t_set_gt = take_Rts_2_world(ref_gt)

    # Compute norms of camera positions
    norms_vm = [np.linalg.norm(pos) for pos in t_set_vm.values()]
    norms_gt = [np.linalg.norm(pos) for pos in t_set_gt.values()]

    # Find average norm
    ave_norm_vm = np.average(norms_vm)
    ave_norm_gt = np.average(norms_gt)

    # Compute the scale factor
    if ave_norm_vm != 0:
        scale_factor = ave_norm_gt / ave_norm_vm
        for key in view_matrices.keys():
            view_matrices[key][:3, 3] = -R_set[key] @ (t_set_vm[key] * scale_factor)
    else:
        print("\t\tNot able to scale. Div by zero.")

def align_method_to_gt(view_matrices, ref_gt):

    # Extract rotation components
    rotations_vm_d, _ = take_Rts_2_world(view_matrices)
    rotations_gt_d, _ = take_Rts_2_world(ref_gt)

    rotations_vm = list(rotations_vm_d.values())
    rotations_gt = list(rotations_gt_d.values())

    # Centroids just in case
    centroid_vm = np.mean(rotations_vm, axis=0)
    centroid_gt = np.mean(rotations_gt, axis=0)

    # Compute cross-covariance matrix
    H = np.zeros((3, 3))
    for R_vm, R_gt in zip(rotations_vm, rotations_gt):
        H += np.dot((R_vm - centroid_vm).T, (R_gt - centroid_gt))

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)
    R_optimal = np.dot(Vt.T, U.T)

    # Ensure proper rotation (determinant check)
    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = np.dot(Vt.T, U.T)

    # Align method views
    for key in view_matrices.keys():
        # Extract rotation and translation from view matrices
        R_cs, t_s = get_Rt_2_world(view_matrices[key])
        R_ws = R_cs.T
        R_ws_aligned = R_optimal @ R_ws

        view_matrices[key][:3, :3] = R_ws_aligned.T  # Transpose back for view matrix

def scale_align_method_poses(method_poses, ground_truth_poses = None):

    normalize_rotation(method_poses)
    normalize_p_to_unit_sphere(method_poses)
    move_centroid_to_origin(method_poses)

    if ground_truth_poses:

        normalize_rotation(ground_truth_poses)
        normalize_p_to_unit_sphere(ground_truth_poses)
        move_centroid_to_origin(ground_truth_poses)

        # Align to GT
        scale_method_to_gt(method_poses, ground_truth_poses)
        align_method_to_gt(method_poses, ground_truth_poses)

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

    ate = compute_ATE(other_poses, ground_truth_poses)
    rpe_trans, rpe_rot = compute_rpe(other_poses, ground_truth_poses)

    eval_metrics = {"ate": ate, "rpe_trans": rpe_trans, "rpe_rot": rpe_rot}
    return pd.DataFrame(comparison_results), eval_metrics


def normalize_image_id(image_id):
    """
    Normalize image ID format.

    Args:
        image_id (str): Original image ID.

    Returns:
        str: Normalized image ID.
    """
    return image_id.zfill(4) if image_id.isdigit() else os.path.splitext(image_id)[0]


def convert_translations(view_matrices):
    for key, M_wc in view_matrices.items():
        # Extract the rotation matrix and translation vector
        R = M_wc[:3, :3]  # Rotation matrix
        t = M_wc[:3, 3]

        M_wc[:3, 3] = -R.T @ t


def transpose_rotations(view_matrices):
    for key, M_wc in view_matrices.items():
        # Extract the rotation matrix and translation vector
        R = M_wc[:3, :3]  # Rotation matrix

        # Transpose the rotation matrix
        M_wc[:3, :3] = R.T

# Function to invert matrices
def invert_matrices(dictionary):
    for key, matrix in dictionary.items():
        # Calculate the inverse
        dictionary[key] = np.linalg.inv(matrix)

def fix_camera_orientation(dictionary):

    # Define a 180-degree rotation around the X-axis
    flip_x = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    for key, matrix in dictionary.items():
        # Apply the flip to the world-to-camera matrix
        dictionary[key] = flip_x @ matrix

def main():

    object_folders = ['AP13', 'MDF14', 'ShSu12', 'SMu1']
    eval_metrics = {}
    method_poses_files = {'COLMAP' : "data/ALGO_poses_result/{}/{}/camera_positions.csv",
                          'CF3DGS' : "data/ALGO_poses_result/{}/HO3Dv3_{}/poses_pred.csv",
                          'GAUSSIAN_OBJECT' : "data/ALGO_poses_result/{}/{}/camera_poses.csv"}
    method_output_names = {'COLMAP' : "colmap",
                           'CF3DGS' : "colmap_free3dgs",
                           'GAUSSIAN_OBJECT' : "gaussian_object"}

    for method, input_file in method_poses_files.items():
        print(f"{method}")
        eval_metrics[method] = {'ate':[],'rpe_trans':[],'rpe_rot':[]}
        for object_folder in object_folders:

            # Input files
            method_path = input_file.format(method, object_folder)
            ground_truth_path = f'data/poses_output/{object_folder}/ground_truth_camera_poses.csv'

            # Load pose
            ground_truth_poses = load_camera_poses(ground_truth_path, 'Image ID', 'Camera Pose', normalize_image_id)
            if(method == "CF3DGS"):
                method_poses = load_camera_poses(method_path, 'Image ID', 'Camera Pose', normalize_image_id, multi_column=True)
            else:
                method_poses = load_camera_poses(method_path, 'Image Name', 'Camera Pose', normalize_image_id)

            if(method == "GAUSSIAN_OBJECT"):
                invert_matrices(method_poses)

            # Filter each dictionary
            common_ids = set(method_poses.keys()) & set(ground_truth_poses.keys())
            gt_filtered = {img_id: copy.deepcopy(ground_truth_poses[img_id]) for img_id in common_ids}
            method_poses = {img_id: copy.deepcopy(method_poses[img_id]) for img_id in common_ids}

            # Sort
            gt_filtered = {key: gt_filtered[key] for key in sorted(gt_filtered)}
            method_poses = {key: method_poses[key] for key in sorted(method_poses)}

            # Normalize & align
            method_poses_ = method_poses.copy()
            scale_align_method_poses(method_poses_)
            scale_align_method_poses(method_poses, gt_filtered)

            # Compare
            method_, _ = compare_camera_poses(method_poses_, method_poses_)
            method_comp, eval_point = compare_camera_poses(gt_filtered, method_poses)

            eval_metrics[method]['ate'].append(eval_point['ate'])
            eval_metrics[method]['rpe_trans'].append(eval_point['rpe_trans'])
            eval_metrics[method]['rpe_rot'].append(eval_point['rpe_rot'])

            # Output Results
            outputFolder = f'visualization/pose_data_comparation/{object_folder}/'
            os.makedirs(outputFolder, exist_ok=True)
            method_comp.to_csv(f'{outputFolder}/{method_output_names[method]}_comparison.csv', index=False)
            method_.to_csv(f'{outputFolder}/{method_output_names[method]}_.csv', index=False)

        outline = f"\t\tAverage Results ATE = {np.average(eval_metrics[method]['ate'])}, "
        outline += f"RPE_trans = {np.average(eval_metrics[method]['rpe_trans'])}, "
        outline += f"RPE_rot = {np.average(eval_metrics[method]['rpe_rot'])}"
        print(outline)

if __name__ == "__main__":
    main()

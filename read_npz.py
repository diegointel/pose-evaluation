import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

import json
data = None

# Function to get the ground truth camera pose for a given image index
def get_camera_pose(index):
    Rt_name = f'Rt_{index:04d}'
    scale_mat_name = f'scale_mat_{index:04d}'
    
    if Rt_name in data.files:
        Rt = data[Rt_name]
    else:
        return None
    
    if scale_mat_name in data.files:
        scale_mat = data[scale_mat_name]
        # Apply the scale matrix to the extrinsic matrix if necessary
        camera_pose = np.dot(scale_mat, np.vstack((Rt, [0, 0, 0, 1])))
    else:
        camera_pose = np.vstack((Rt, [0, 0, 0, 1]))
    
    return camera_pose

# Function to plot camera poses in 3D space
def plot_camera_poses(data, object_position, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for array_name in data.files:
        if array_name.startswith('Rt_'):
            index = int(array_name.split('_')[1])
            camera_pose = get_camera_pose(index)
            if camera_pose is not None:
                # Extract the translation part of the camera pose
                translation = camera_pose[:3, 3]
                # Extract the rotation part of the camera pose
                rotation = camera_pose[:3, :3]
                # Define the direction vector (e.g., the z-axis of the camera)
                direction = rotation @ np.array([0, 0, 1])
                # Plot the camera position
                ax.scatter(translation[0], translation[1], translation[2], c='b')
                # Plot the camera orientation as an arrow
                ax.quiver(translation[0], translation[1], translation[2],
                          direction[0], direction[1], direction[2], length=0.1, normalize=True, color='r')
    
    # Plot the object position
    ax.scatter(object_position[0], object_position[1], object_position[2], c='g', s=100, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Camera Poses in 3D Space')
    plt.savefig(save_path)
    plt.show()

# Function to save camera poses to a CSV file
def save_camera_poses_to_csv(data, csv_path):
    print(data)
    npz_dict = {key: data[key].tolist() for key in data}
    dk = ""
    dk += json.dumps(npz_dict)
    with open(csv_path + ".json", mode='w', newline='') as file:
        file.write(dk)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image ID', 'Camera Pose'])
        
        for array_name in data.files:
            
            if array_name.startswith('Rt_'):
                index = int(array_name.split('_')[1])
                camera_pose = get_camera_pose(index)
                if camera_pose is not None:
                    writer.writerow([index, camera_pose.flatten()])
    

data_folders = ['AP13', 'MDF14_ori', 'ShSu12_ori', 'SMu1_ori']
for folder in data_folders:
    # Load the .npz file
    file_path = f'data/HO3Dv3/{folder}/cameras_sphere.npz'
    data = np.load(file_path)

    # Define the object position (e.g., at the origin)
    object_position = np.array([0, 0, 0])
    folder_dir = f'data/poses_output/{folder}'
    os.makedirs(folder_dir, exist_ok=True)
    # Plot and save the camera poses
    plot_camera_poses(data, object_position, f'{folder_dir}/camera_poses.png')

    # Save the camera poses to a CSV file
    save_camera_poses_to_csv(data, f'{folder_dir}/ground_truth_camera_poses.csv')
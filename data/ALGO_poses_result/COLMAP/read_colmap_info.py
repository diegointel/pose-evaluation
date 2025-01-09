import os
import sqlite3
import numpy as np
import pycolmap

def extract_camera_positions(sparse_folder, database_path, output_csv):
    # Read the cameras, images, and points3D from the sparse folder
    reconstruction = pycolmap.Reconstruction(os.path.join(sparse_folder, '0'))
    images = reconstruction.images

    # Connect to the database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Query the database to get the image names
    cursor.execute("SELECT image_id, name FROM images")
    image_names = {row[0]: row[1] for row in cursor.fetchall()}

    # Extract camera positions
    camera_positions = []
    camera_poses = []
    for image_id, image in images.items():
        R = image.cam_from_world.rotation.matrix().real
        t = image.cam_from_world.translation
        camera_position = -R.T @ t
        camera_positions.append((image_names[image_id], camera_position))
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t
        camera_poses.append((image_names[image_id], camera_pose))

    # Save the camera positions to a CSV file
    with open(output_csv, 'w') as f:
        f.write('Image Name,Camera Position\n')
        for image_name, camera_position in camera_positions:
            f.write(f'{image_name},{camera_position[0]},{camera_position[1]},{camera_position[2]}\n')

    with open(output_csv, 'w') as f:
        f.write('Image Name,Camera Pose\n')
        for image_name, camera_pose in camera_poses:
            pose_str = ' '.join(map(str, camera_pose.flatten()))
            f.write(f'{image_name},"{pose_str}"\n')

    print(f'Camera positions saved to {output_csv}')

# Example usage
data_folders = ['AP13', 'MDF14', 'ShSu12', 'SMu1']
for folder in data_folders:
    sparse_folder = f'{folder}/sparse'
    database_path = f'{folder}/database.db'
    output_csv = f'{folder}/camera_positions.csv'
    extract_camera_positions(sparse_folder, database_path, output_csv)
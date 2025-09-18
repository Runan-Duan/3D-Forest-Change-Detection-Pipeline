import os
import open3d as o3d
import numpy as np
import copy
from registration import *



# Load clouds
print("Loading point clouds...")
source_path = r"data/20240607_UAV-Photo_Sandhausen_5cm.ply"
target_path = r"data/20250613_UAV-Photo_Sandhausen_5cm.ply"

source = o3d.io.read_point_cloud(source_path, format='ply')
target = o3d.io.read_point_cloud(target_path, format='ply')

print("Original Source Points:", len(source.points))
print("Original Target Points:", len(target.points))

# Step 1: Coarse global registration
coarse_voxel = 0.20
source_down, source_fpfh = preprocess_point_cloud(source, coarse_voxel)
target_down, target_fpfh = preprocess_point_cloud(target, coarse_voxel)

global_result = global_registration(source_down, target_down,
                                    source_fpfh, target_fpfh,
                                    coarse_voxel)
print("Coarse Registration Result:")
print(f"Fitness: {global_result.fitness}, Inlier RMSE: {global_result.inlier_rmse}")
print("Transformation (Coarse):\n", global_result.transformation)


# Step 2: Multi-scale Colored ICP refinement
final_transformation = local_registration_multiscale(
    source, target, global_result.transformation,
    voxel_sizes=[0.20, 0.10, 0.05]
)

print("\nFinal Registration Result:")
print("Transformation (Refined):\n", final_transformation)

# Step 3: Save transformed full-resolution cloud
source_registered = copy.deepcopy(source)
source_registered.transform(final_transformation)
o3d.io.write_point_cloud("registered_forest_scan.ply", source_registered)
print("Registered point cloud saved to 'registered_forest_scan.ply'.")

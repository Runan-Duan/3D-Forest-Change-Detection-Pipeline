import os
import open3d as o3d
import numpy as np
import copy
import argparse

def local_registration_multiscale(source, target, init_transformation, voxel_sizes=[0.2, 0.1, 0.05]):
    """
    Multi-scale Colored ICP refinement.
    """
    current_transformation = init_transformation

    for i, voxel_size in enumerate(voxel_sizes):
        print(f"\nRefinement stage {i+1} with voxel_size={voxel_size} ...")
        distance_threshold = voxel_size * 1.0

        # Downsample for this scale
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # Estimate normals
        radius_normal = voxel_size * 2.0
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

        # Colored ICP
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )

        print(f"  Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
        current_transformation = result.transformation

    return current_transformation


# Load point clouds
print("Loading point clouds...")
source_path = r"../data/20240607_ground.ply"
target_path = r"../data/20250613_ground.ply"

source = o3d.io.read_point_cloud(source_path, format='ply')
target = o3d.io.read_point_cloud(target_path, format='ply')

print("Original Source Points:", len(source.points))
print("Original Target Points:", len(target.points))

# Start with identity transformation (already coarsely aligned)
initial_transformation = np.eye(4)  # Identity matrix - assuming clouds are already coarsely aligned
colored_transformation = local_registration_multiscale(source, target, initial_transformation)
print("Colored ICP Transformation:")
print(colored_transformation)

# Apply final transformation
source_registered = copy.deepcopy(source)
source_registered.transform(colored_transformation)

# Evaluate final result
evaluation = o3d.pipelines.registration.evaluate_registration(
    source_registered, target, 0.05
)
print(f"\nFinal Evaluation:")
print(f"Fitness: {evaluation.fitness:.4f}")
print(f"Inlier RMSE: {evaluation.inlier_rmse:.4f}")

# Save registered point cloud
output_path = "registered_forest_scan.ply"
o3d.io.write_point_cloud(output_path, source_registered)
print(f"\nRegistered point cloud saved to '{output_path}'")

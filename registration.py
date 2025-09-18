import os
import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess a point cloud by downsampling and computing FPFH features.
    """
    print(f"Preprocessing with voxel_size={voxel_size} ...")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.5
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def global_registration(source_down, target_down, source_fpfh,
                        target_fpfh, voxel_size):
    """
    Coarse global registration using RANSAC on FPFH features.
    """
    print("Running global registration (RANSAC)...")
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )
    return result

def local_registration_multiscale(source, target, init_transformation, voxel_sizes=[0.2, 0.1, 0.05]):
    """
    Multi-scale Colored ICP refinement.
    """
    current_transformation = init_transformation

    for i, voxel_size in enumerate(voxel_sizes):
        print(f"\nRefinement stage {i+1} with voxel_size={voxel_size} ...")
        distance_threshold = voxel_size * 0.4

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
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        print(f"  Fitness: {result.fitness:.4f}, Inlier RMSE: {result.inlier_rmse:.4f}")
        current_transformation = result.transformation

    return current_transformation

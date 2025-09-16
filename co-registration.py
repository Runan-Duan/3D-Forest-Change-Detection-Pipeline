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

def visualize(source, target, transformation):
    """
    Visualize alignment result.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])   # Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7735, 2.3625, 0.8247],
                                      up=[0.3105, -0.5878, -0.7468])

def main():
    # Load clouds (already voxelized to 5 cm in PDAL)
    print("Loading point clouds...")
    source_path = "data/20240607_UAV-Photo_Sandhausen_5cm.ply"
    target_path = "data/20250613_UAV-Photo_Sandhausen_5cm.ply"

    source = o3d.io.read_point_cloud(source_path, format='ply')
    target = o3d.io.read_point_cloud(target_path, format='ply')

    print("Original Source Points:", len(source.points))
    print("Original Target Points:", len(target.points))

    # Coarse global registration 
    coarse_voxel = 0.20  # use larger voxel 20 cm
    source_down, source_fpfh = preprocess_point_cloud(source, coarse_voxel)
    target_down, target_fpfh = preprocess_point_cloud(target, coarse_voxel)

    global_result = global_registration(source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        coarse_voxel)
    print("Coarse Registration Result:")
    print(f"Fitness: {global_result.fitness}, Inlier RMSE: {global_result.inlier_rmse}")
    print("Transformation (Coarse):\n", global_result.transformation)

    visualize(source, target, global_result.transformation)

    # Multi-scale Colored ICP refinement 
    final_transformation = local_registration_multiscale(
        source, target, global_result.transformation,
        voxel_sizes=[0.20, 0.10, 0.05]
    )

    print("\nFinal Registration Result:")
    print("Transformation (Refined):\n", final_transformation)

    visualize(source, target, final_transformation)

    # Save transformed full-resolution cloud
    source_registered = copy.deepcopy(source)
    source_registered.transform(final_transformation)
    o3d.io.write_point_cloud("registered_forest_scan.ply", source_registered)
    print("Registered point cloud saved to 'registered_forest_scan.ply'.")

if __name__ == "__main__":
    main()

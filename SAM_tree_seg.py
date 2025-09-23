# ==============================
# SAM Pipeline for Tree Detection with Patch Handling
# ==============================
import rioxarray as rxr
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import laspy
from scipy.spatial import cKDTree
import rasterio.features
from rasterio.warp import Resampling
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch

# ==============================
# Data Loading and Alignment Functions
# ==============================
def load_and_align_data():
    """Load and align RGB and DSM data"""
    print("Loading and aligning data...")
    
    # Orthophoto RGB and DSM
    rgb_2024 = rxr.open_rasterio("data/20240607_UAV-Photo_Sandhausen_ortho_5cm_clipped.tif")
    rgb_2025 = rxr.open_rasterio("data/20250613_UAV-Photo_Sandhausen_ortho_5cm_clipped.tif")
    dsm_2024 = rxr.open_rasterio("data/20240607_UAV-Photo_Sandhausen_dem_5cm_clipped.tif")
    dsm_2025 = rxr.open_rasterio("data/20250613_UAV-Photo_Sandhausen_dem_clipped.tif")
    
    # Reproject 2025 data to match 2024
    rgb_2025_aligned = rgb_2025.rio.reproject_match(rgb_2024, resampling=Resampling.bilinear)
    dsm_2025_aligned = dsm_2025.rio.reproject_match(dsm_2024, resampling=Resampling.bilinear)
    
    # Create geometry from 2024 DSM bounds for cropping
    bounds = dsm_2024.rio.bounds()
    crop_geom = box(*bounds)
    
    # Crop 2025 DSM by 2024 DSM extent
    dsm_2025_cropped = dsm_2025_aligned.rio.clip([crop_geom], drop=True)
    
    # Mask by actual data values (not just extent)
    mask_2024 = ~dsm_2024.isnull()
    dsm_2025_cropped = dsm_2025_cropped.where(mask_2024)
    
    return rgb_2024, rgb_2025_aligned, dsm_2024, dsm_2025_cropped

def convert_to_numpy_arrays(rgb_2024, rgb_2025_aligned, dsm_2024, dsm_2025_cropped):
    """Convert rioxarray data to numpy arrays"""
    print("Converting to numpy arrays...")
    
    # Extract numpy arrays
    rgb_2024_np = rgb_2024[-3:, :, :].values
    rgb_2025_np = rgb_2025_aligned[-3:, :, :].values
    
    dsm_2024_np = dsm_2024.values
    dsm_2025_np = dsm_2025_cropped.values
    
    # Transpose RGB to (H, W, C)
    rgb_2024_np = np.transpose(rgb_2024_np, (1, 2, 0))
    rgb_2025_np = np.transpose(rgb_2025_np, (1, 2, 0))
    
    # Expand DSM to (H, W, C)
    dsm_2024_np = np.transpose(dsm_2024_np, (1, 2, 0))
    dsm_2025_np = np.transpose(dsm_2025_np, (1, 2, 0))
    
    # Create 4-channel input (RGB + DSM)
    input_2024 = np.concatenate([rgb_2024_np, dsm_2024_np], axis=-1)
    input_2025 = np.concatenate([rgb_2025_np, dsm_2025_np], axis=-1)
    
    print(f"Input 2024 shape: {input_2024.shape}")
    print(f"Input 2025 shape: {input_2025.shape}")
    
    return input_2024, input_2025

# ==============================
# Tree-Focused Height Intensity Preprocessing
# ==============================
def prepare_tree_focused_input(rgbd_input, enhance_dead_trees=True):
    """
    Create height-intensity input optimized for tree detection
    Input: 4-channel array (H, W, 4) where channels are [R, G, B, DSM]
    """
    print("Preparing tree-focused input with height intensity...")
    
    # Extract RGB and DSM from 4-channel input
    rgb_array = rgbd_input[:, :, :3].astype(np.float32)
    dsm_array = rgbd_input[:, :, 3].astype(np.float32)
    
    # Handle NaN values in DSM
    valid_dsm = ~np.isnan(dsm_array)
    
    if np.sum(valid_dsm) > 0:
        # Robust normalization focusing on tree heights
        dsm_valid_values = dsm_array[valid_dsm]
        
        # Use percentiles to focus on vegetation heights
        p5, p50, p95 = np.percentile(dsm_valid_values, [5, 50, 95])
        
        # Normalize with emphasis on tree height range (above median)
        tree_height_range = p95 - p50
        if tree_height_range > 0:
            # Create height factor emphasizing taller objects (trees)
            height_factor = np.zeros_like(dsm_array)
            height_factor[valid_dsm] = np.clip(
                (dsm_array[valid_dsm] - p50) / tree_height_range, 0, 1
            )
            
            # Additional boost for very tall objects (likely trees)
            height_factor = np.power(height_factor, 0.7)  # Gamma correction
            height_factor = height_factor * 0.8 + 0.6  # Scale to 0.6-1.4 range
        else:
            height_factor = np.ones_like(dsm_array)
        
        # Fill invalid areas with neutral factor
        height_factor[~valid_dsm] = 1.0
    else:
        height_factor = np.ones_like(dsm_array)
    
    # Apply height intensity modulation
    enhanced_rgb = rgb_array.copy()
    for c in range(3):
        enhanced_rgb[:, :, c] = rgb_array[:, :, c] * height_factor
    
    # Enhance dead trees (low RGB values but high DSM)
    if enhance_dead_trees:
        # Detect potential dead trees: low vegetation index + high height
        red = rgb_array[:, :, 0]
        green = rgb_array[:, :, 1]
        blue = rgb_array[:, :, 2]
        
        # Simple vegetation index
        veg_index = (green - red) / (green + red + 1e-8)
        
        # Dead tree mask: low vegetation + significant height
        dead_tree_mask = (veg_index < 0.1) & (height_factor > 1.2) & valid_dsm
        
        # Enhance contrast for dead trees
        for c in range(3):
            enhanced_rgb[dead_tree_mask, c] *= 1.3
    
    # Ensure valid range and convert to uint8
    enhanced_rgb = np.clip(enhanced_rgb, 0, 255).astype(np.uint8)
    
    return enhanced_rgb

# ==============================
# Patch-based Processing for Large Images
# ==============================
class PatchProcessor:
    def __init__(self, patch_size=1024, overlap=128):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def create_patches(self, image):
        """Create overlapping patches from large image"""
        h, w = image.shape[:2]
        patches = []
        patch_info = []
        
        for y in range(0, h - self.overlap, self.stride):
            for x in range(0, w - self.overlap, self.stride):
                # Ensure don't exceed image bounds
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)
                
                patch = image[y_start:y_end, x_start:x_end]
                
                # Ensure patch is exactly patch_size x patch_size
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    # Pad if necessary
                    pad_h = self.patch_size - patch.shape[0]
                    pad_w = self.patch_size - patch.shape[1]
                    
                    if patch.ndim == 3:
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), 
                                     mode='reflect')
                    else:
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), 
                                     mode='reflect')
                
                patches.append(patch)
                patch_info.append({
                    'y_start': y_start, 'y_end': y_end,
                    'x_start': x_start, 'x_end': x_end,
                    'original_shape': (y_end - y_start, x_end - x_start)
                })
        
        return patches, patch_info
    
    def merge_patch_masks(self, patch_masks, patch_info, original_shape):
        """Merge masks from patches back to original image"""
        h, w = original_shape[:2]
        merged_masks = []
        
        # Collect all masks with global coordinates
        global_masks = []
        
        for patch_idx, (masks, info) in enumerate(zip(patch_masks, patch_info)):
            for mask_dict in masks:
                mask = mask_dict['segmentation']
                
                # Skip if mask is at patch boundary
                if self._is_boundary_mask(mask, info['original_shape']):
                    continue
                
                # Convert to global coordinates
                global_mask = np.zeros((h, w), dtype=bool)
                mask_h, mask_w = info['original_shape']
                global_mask[info['y_start']:info['y_start']+mask_h, 
                           info['x_start']:info['x_start']+mask_w] = mask[:mask_h, :mask_w]
                
                global_masks.append({
                    'segmentation': global_mask,
                    'area': mask_dict['area'],
                    'stability_score': mask_dict['stability_score'],
                    'predicted_iou': mask_dict['predicted_iou'],
                    'point_coords': mask_dict['point_coords'],
                    'bbox': [info['x_start'] + mask_dict['bbox'][0],
                            info['y_start'] + mask_dict['bbox'][1],
                            mask_dict['bbox'][2], mask_dict['bbox'][3]]
                })
        
        # Remove overlapping masks
        return self._remove_overlapping_masks(global_masks)
    
    def _is_boundary_mask(self, mask, patch_shape, boundary_threshold=10):
        """Check if mask touches patch boundary"""
        h, w = patch_shape
        
        # Check if mask extends to patch edges
        touches_edge = (
            np.any(mask[:boundary_threshold, :]) or  # Top edge
            np.any(mask[-boundary_threshold:, :]) or  # Bottom edge
            np.any(mask[:, :boundary_threshold]) or  # Left edge
            np.any(mask[:, -boundary_threshold:])     # Right edge
        )
        
        return touches_edge
    
    def _remove_overlapping_masks(self, masks, iou_threshold=0.5):
        """Remove overlapping masks, keeping higher quality ones"""
        if len(masks) <= 1:
            return masks
        
        # Sort by quality score
        masks_sorted = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        
        keep_masks = []
        for i, mask1 in enumerate(masks_sorted):
            should_keep = True
            
            for mask2 in keep_masks:
                iou = self._calculate_mask_iou(mask1['segmentation'], mask2['segmentation'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_masks.append(mask1)
        
        return keep_masks
    
    def _calculate_mask_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0
        return intersection / union

# ==============================
# Tree SAM with Patch Support
# ==============================
class TreeSAM:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
        print(f"Loading SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # Tree-optimized parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=24,              # Balanced detail vs speed
            pred_iou_thresh=0.7,             # High quality threshold
            stability_score_thresh=0.8,       # Stable predictions only
            crop_n_layers=1,                 # Single crop layer
            crop_n_points_downscale_factor=2,
            min_mask_region_area=50,         # Minimum tree crown area
            box_nms_thresh=0.7,              # Reduce duplicate detections
        )
        
        self.patch_processor = PatchProcessor(patch_size=1024, overlap=128)
    
    def generate_tree_masks(self, image, use_patches=True):
        """Generate tree masks with optional patch processing"""
        print(f"Image shape: {image.shape}")
        
        # Check if need patch processing
        if use_patches and (image.shape[0] > 1024 or image.shape[1] > 1024):
            print("Using patch-based processing for large image...")
            return self._generate_masks_with_patches(image)
        else:
            print("Processing image as single patch...")
            # Ensure image is correct format for SAM
            if image.shape[0] != 1024 or image.shape[1] != 1024:
                print(f"Resizing image from {image.shape[:2]} to (1024, 1024) for SAM")
                image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            
            masks = self.mask_generator.generate(image)
            print(f"Generated {len(masks)} masks")
            return masks
    
    def _generate_masks_with_patches(self, image):
        """Process large image using patches"""
        patches, patch_info = self.patch_processor.create_patches(image)
        patch_masks = []
        
        print(f"Processing {len(patches)} patches...")
        for i, patch in enumerate(patches):
            print(f"Processing patch {i+1}/{len(patches)}")
            
            # Ensure patch is correct size
            assert patch.shape[:2] == (1024, 1024), f"Patch shape {patch.shape} not (1024, 1024)"
            
            masks = self.mask_generator.generate(patch)
            patch_masks.append(masks)
        
        # Merge patches
        print("Merging patch results...")
        merged_masks = self.patch_processor.merge_patch_masks(
            patch_masks, patch_info, image.shape
        )
        
        print(f"Merged to {len(merged_masks)} total masks")
        return merged_masks

# ==============================
# Enhanced Tree Filtering
# ==============================
def filter_tree_masks(masks, min_area=100, max_area=8000, min_stability=0.7):
    """Filter masks for tree characteristics"""
    print("Filtering masks for trees...")
    
    filtered_masks = []
    
    for mask_dict in masks:
        # Basic filters
        area = mask_dict.get('area', np.sum(mask_dict['segmentation']))
        stability = mask_dict.get('stability_score', 1.0)
        
        if not (min_area <= area <= max_area):
            continue
        
        if stability < min_stability:
            continue
        
        # Tree shape analysis
        mask = mask_dict['segmentation']
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            
            # Compactness (trees are roughly circular)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                if compactness < 0.2:  # Very elongated shapes unlikely to be trees
                    continue
            
            # Aspect ratio
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3:  # Very elongated
                    continue
        
        mask_dict['area'] = area
        mask_dict['stability_score'] = stability
        filtered_masks.append(mask_dict)
    
    print(f"Filtered to {len(filtered_masks)} tree candidates")
    return filtered_masks

# ==============================
# Complete Pipeline
# ==============================
def run_tree_detection_pipeline():
    """Complete tree detection pipeline"""
    # Load and align data using your exact implementation
    rgb_2024, rgb_2025_aligned, dsm_2024, dsm_2025_cropped = load_and_align_data()
    
    # Get spatial reference info
    transform_2024 = rgb_2024.rio.transform()
    crs_2024 = rgb_2024.rio.crs
    
    # Convert to numpy arrays using your exact implementation
    input_2024, input_2025 = convert_to_numpy_arrays(
        rgb_2024, rgb_2025_aligned, dsm_2024, dsm_2025_cropped
    )
    
    # Prepare tree-focused inputs (convert 4-channel to 3-channel for SAM)
    print("\nPreparing 2024 data for SAM...")
    sam_input_2024 = prepare_tree_focused_input(input_2024, enhance_dead_trees=True)
    
    print("Preparing 2025 data for SAM...")
    sam_input_2025 = prepare_tree_focused_input(input_2025, enhance_dead_trees=True)
    
    # Initialize SAM
    tree_sam = TreeSAM()
    
    # Generate masks
    print("\nGenerating 2024 masks...")
    masks_2024 = tree_sam.generate_tree_masks(sam_input_2024)
    
    print("Generating 2025 masks...")
    masks_2025 = tree_sam.generate_tree_masks(sam_input_2025)
    
    # Filter for trees
    print("\nFiltering masks...")
    tree_masks_2024 = filter_tree_masks(masks_2024)
    tree_masks_2025 = filter_tree_masks(masks_2025)
    
    print(f"\nFinal results:")
    print(f"2024: {len(tree_masks_2024)} trees detected")
    print(f"2025: {len(tree_masks_2025)} trees detected")
    
    return tree_masks_2024, tree_masks_2025, transform_2024, crs_2024, input_2024, input_2025

# ==============================
# Mask to GeoDataFrame Conversion
# ==============================
def masks_to_geodataframe(masks, transform, crs):
    """Convert masks to GeoDataFrame"""
    print("Converting masks to GeoDataFrame...")
    
    polygons = []
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        
        # Convert to polygons
        shapes_gen = rasterio.features.shapes(mask.astype(np.uint8), transform=transform)
        
        for geom, val in shapes_gen:
            if val == 1:
                coords = geom['coordinates'][0]
                if len(coords) >= 4:
                    poly = Polygon(coords)
                    if poly.is_valid and poly.area > 0:
                        polygons.append({
                            'geometry': poly,
                            'mask_id': i,
                            'area_pixels': mask_dict.get('area', np.sum(mask)),
                            'area_m2': poly.area,
                            'stability_score': mask_dict.get('stability_score', 0),
                            'predicted_iou': mask_dict.get('predicted_iou', 0)
                        })
    
    return gpd.GeoDataFrame(polygons, crs=crs)


if __name__ == "__main__":
    # Run pipeline
    masks_2024, masks_2025, transform, crs, input_2024, input_2025 = run_tree_detection_pipeline()
    
    # Convert to GeoDataFrames
    gdf_2024 = masks_to_geodataframe(masks_2024, transform, crs)
    gdf_2025 = masks_to_geodataframe(masks_2025, transform, crs)
    
    # Save results
    gdf_2024.to_file("trees_2024_enhanced.geojson", driver="GeoJSON")
    gdf_2025.to_file("trees_2025_enhanced.geojson", driver="GeoJSON")
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to trees_2024_enhanced.geojson and trees_2025_enhanced.geojson")
    print(f"\nData shapes:")
    print(f"Input 2024: {input_2024.shape} (4-channel: RGB+DSM)")
    print(f"Input 2025: {input_2025.shape} (4-channel: RGB+DSM)")
    print(f"Trees detected - 2024: {len(gdf_2024)}, 2025: {len(gdf_2025)}")
    
    # Statistics
    if len(gdf_2024) > 0:
        print(f"\n2024 Tree Statistics:")
        print(f"Average area: {gdf_2024['area_m2'].mean():.1f} m²")
        print(f"Average stability score: {gdf_2024['stability_score'].mean():.3f}")
    
    if len(gdf_2025) > 0:
        print(f"\n2025 Tree Statistics:")
        print(f"Average area: {gdf_2025['area_m2'].mean():.1f} m²")
        print(f"Average stability score: {gdf_2025['stability_score'].mean():.3f}")
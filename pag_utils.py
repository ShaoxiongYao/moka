from PIL import Image, ImageDraw, ImageFont
import numpy as np
import open3d as o3d

import os
import subprocess
import matplotlib.pyplot as plt

def plot_2d_points(points_cropped, context):
    """Plot 2D points from context dictionary on the image."""
    plt.figure(figsize=(10, 8))
    
    # Show the image
    plt.imshow(points_cropped[:, :, 2])
    
    # Plot target point
    if context['keypoints_2d']['target'] is not None:
        target = context['keypoints_2d']['target']
        plt.plot(target[0], target[1], 'ro', markersize=10, label='Target')
    
    # Plot pre-contact waypoints
    for i, point in enumerate(context['waypoints_2d']['pre_contact']):
        plt.plot(point[0], point[1], 'go', markersize=8, label='Pre-contact' if i == 0 else '')
    
    # Plot post-contact waypoints
    for i, point in enumerate(context['waypoints_2d']['post_contact']):
        plt.plot(point[0], point[1], 'bo', markersize=8, label='Post-contact' if i == 0 else '')
    
    plt.legend()
    plt.title('2D Keypoints and Waypoints')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_point_array_from_rgbd(rgb_img, depth_img, intrinsic):
    """
    Creates an H x W x 3 array of 3D points from RGB and Depth numpy arrays.
    
    Returns:
        points: numpy array of shape (H, W, 3) containing (x, y, z) coordinates
    """
    H, W = depth_img.shape[:2]
    
    # Extract intrinsic parameters
    fx = intrinsic['fx']
    fy = intrinsic['fy']
    cx = intrinsic['cx']
    cy = intrinsic['cy']
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Get depth values
    z = depth_img.astype(np.float32)
    
    # Apply depth truncation (optional, matching Open3D's depth_trunc=3.0)
    z = np.where(z > 3.0, 0, z)
    
    # Back-project to 3D using pinhole camera model
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into H x W x 3 array
    points = np.stack([x, y, z], axis=-1)
    
    # Optional: Apply the same coordinate transformation as in commented line
    # points[..., 1] *= -1  # Flip y
    # points[..., 2] *= -1  # Flip z
    
    return points

def process_objects(all_object_names, data_dir, img_path):
    """
    Run GSAM2 and mask extraction for all objects in the list.
    
    Args:
        all_object_names: List of object names to process
        data_dir: Output directory path
        img_path: Path to input image
    """
    
    # Set working directory to match bash script behavior
    working_dir = '/home/ydu/haowen/moka'
    
    # Extract base filename from img_path to construct json_path
    img_basename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "camera1_rgb_cropped"
    json_path = os.path.join(data_dir, f"{img_basename}_gsam2.json")
    
    # Extract directory name for output prefix (e.g., "test_moka_1761943839.687097")
    dir_name = os.path.basename(os.path.normpath(data_dir))
    output_prefix = os.path.join(data_dir, dir_name)  # Full path prefix
    
    for object_name in all_object_names:
        # Build object prompt with period (matching bash script format)
        object_prompt = f"{object_name}."
        
        print(f"Processing: {object_prompt}")
        
        # Run GSAM2 with shell=True and proper working directory
        command_gsam2 = f'python ../real2sim/run_gsam2.py \
                          --text-prompt "{object_prompt}" \
                          --img-path {img_path} --output-dir {data_dir}'
        
        subprocess.run(
            command_gsam2,
            shell=True,
            cwd=working_dir,
            check=True
        )
        
        # Run mask extraction with proper output prefix
        subprocess.run(
            [
                "python", "../real2sim/mask_extraction.py",
                "--json", json_path,
                "--output", output_prefix,  # Changed: use prefix instead of directory
                "--png"
            ],
            cwd=working_dir,
            check=True
        )
        
        print(f"Completed: {object_name}")



def annotate_keypoints(image, data, radius=10):
    """
    Annotate keypoints and waypoints on a PIL image.
    
    Args:
        image: PIL Image object
        data: Dictionary containing keypoints_2d and waypoints_2d
    
    Returns:
        PIL Image with annotations
    """
    # Create a copy to avoid modifying the original
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Define colors and sizes
    colors = {
        'target': 'red',
        'pre_contact': 'green',
        'post_contact': 'blue'
    }
    
    # Draw target point
    if data['keypoints_2d']['target'] is not None:
        target = data['keypoints_2d']['target']
        x, y = float(target[0]), float(target[1])
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                     fill=colors['target'], outline='white', width=2)
        draw.text((x+radius, y-radius), 'target', fill=colors['target'])
    
    # Draw pre_contact waypoints
    if data['waypoints_2d']['pre_contact']:
        for i, point in enumerate(data['waypoints_2d']['pre_contact']):
            x, y = float(point[0]), float(point[1])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                         fill=colors['pre_contact'], outline='white', width=2)
            draw.text((x+radius, y-radius), f'pre_contact_{i}', fill=colors['pre_contact'])
    
    # Draw post_contact waypoints
    if data['waypoints_2d']['post_contact']:
        for i, point in enumerate(data['waypoints_2d']['post_contact']):
            x, y = float(point[0]), float(point[1])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                         fill=colors['post_contact'], outline='white', width=2)
            draw.text((x+radius, y-radius), f'post_contact_{i}', fill=colors['post_contact'])
    
    return img


def create_sphere_at_point(point, radius=0.01, color=[1, 0, 0]):
    """Create a small sphere marker at a 3D point."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere


def create_line_between_points(point1, point2, color=[0, 1, 0]):
    """Create a line between two 3D points."""
    points = [point1, point2]
    lines = [[0, 1]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set


def get_3d_points_from_backprojected(points_3d_array, context, window_size=10):
    """
    Extract 3D coordinates for keypoints and waypoints from backprojected point array.
    
    Args:
        points_3d_array: (H, W, 3) array of 3D points
        context: Dictionary containing 'keypoints_2d' and 'waypoints_2d'
    
    Returns:
        Dictionary with 3D points for target and waypoints
    """
    
    def get_point_at_pixel(points_array, u, v, window_size=window_size):
        """Get averaged 3D point around a pixel location."""
        u_int = int(round(u))
        v_int = int(round(v))
        
        half_window = window_size // 2
        u_min = max(0, u_int - half_window)
        u_max = min(points_array.shape[1], u_int + half_window + 1)
        v_min = max(0, v_int - half_window)
        v_max = min(points_array.shape[0], v_int + half_window + 1)
        
        # Get patch of 3D points
        points_patch = points_array[v_min:v_max, u_min:u_max]
        
        # Filter out invalid points (where z=0)
        valid_mask = points_patch[..., 2] > 0
        valid_points = points_patch[valid_mask]
        
        if len(valid_points) > 0:
            return np.median(valid_points, axis=0)  # Median is more robust
        else:
            return points_array[v_int, u_int]
    
    # Prepare result dictionary
    result = {
        'keypoints_3d': {},
        'waypoints_3d': {}
    }
    
    # Process target keypoint
    if context['keypoints_2d']['target'] is not None:
        target_2d = context['keypoints_2d']['target']
        target_3d = get_point_at_pixel(points_3d_array, target_2d[0], target_2d[1])
        result['keypoints_3d']['target'] = target_3d
        print(f"Target 2D: {target_2d}, 3D: {target_3d}")
    else:
        result['keypoints_3d']['target'] = None
    
    # Process grasp keypoint
    if context['keypoints_2d']['grasp'] is not None:
        grasp_2d = context['keypoints_2d']['grasp']
        grasp_3d = get_point_at_pixel(points_3d_array, grasp_2d[0], grasp_2d[1])
        result['keypoints_3d']['grasp'] = grasp_3d
        print(f"Grasp 2D: {grasp_2d}, 3D: {grasp_3d}")
    else:
        result['keypoints_3d']['grasp'] = None
    
    # Process function keypoint
    if context['keypoints_2d']['function'] is not None:
        function_2d = context['keypoints_2d']['function']
        function_3d = get_point_at_pixel(points_3d_array, function_2d[0], function_2d[1])
        result['keypoints_3d']['function'] = function_3d
        print(f"Function 2D: {function_2d}, 3D: {function_3d}")
    else:
        result['keypoints_3d']['function'] = None
    
    # Process pre_contact waypoints
    result['waypoints_3d']['pre_contact'] = []
    for waypoint_2d in context['waypoints_2d']['pre_contact']:
        waypoint_3d = get_point_at_pixel(points_3d_array, waypoint_2d[0], waypoint_2d[1])
        result['waypoints_3d']['pre_contact'].append(waypoint_3d)
        print(f"Pre-contact 2D: {waypoint_2d}, 3D: {waypoint_3d}")
    
    # Process post_contact waypoints
    result['waypoints_3d']['post_contact'] = []
    for waypoint_2d in context['waypoints_2d']['post_contact']:
        waypoint_3d = get_point_at_pixel(points_3d_array, waypoint_2d[0], waypoint_2d[1])
        result['waypoints_3d']['post_contact'].append(waypoint_3d)
        print(f"Post-contact 2D: {waypoint_2d}, 3D: {waypoint_3d}")
    
    return result


def visualize_point_cloud_with_keypoints(points_3d_array, rgb_image, points_3d_dict, 
                                         save_path=None):
    """
    Visualize point cloud with keypoints and waypoints overlaid.
    
    Args:
        points_3d_array: (H, W, 3) array of backprojected 3D points
        rgb_image: RGB image (PIL Image or numpy array)
        points_3d_dict: Dictionary from get_3d_points_from_backprojected()
        save_path: Optional path to save the point cloud (e.g., 'pointcloud.ply')
    """
    
    # Convert PIL Image to numpy if needed
    if isinstance(rgb_image, Image.Image):
        rgb_image = np.array(rgb_image)
    
    # Flatten arrays and filter valid points
    points = points_3d_array.reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0
    
    valid_mask = points[:, 2] > 0  # Filter out invalid depth
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Create Open3D point cloud
    print("Creating point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # List to hold all geometries to visualize
    geometries = [pcd]
    
    # Add target keypoint (RED sphere)
    if points_3d_dict['keypoints_3d']['target'] is not None:
        target_sphere = create_sphere_at_point(
            points_3d_dict['keypoints_3d']['target'], 
            radius=0.015, 
            color=[1, 0, 0]
        )
        geometries.append(target_sphere)
    
    # Add grasp keypoint (BLUE sphere)
    if points_3d_dict['keypoints_3d']['grasp'] is not None:
        grasp_sphere = create_sphere_at_point(
            points_3d_dict['keypoints_3d']['grasp'], 
            radius=0.015, 
            color=[0, 0, 1]
        )
        geometries.append(grasp_sphere)
    
    # Add function keypoint (YELLOW sphere)
    if points_3d_dict['keypoints_3d']['function'] is not None:
        function_sphere = create_sphere_at_point(
            points_3d_dict['keypoints_3d']['function'], 
            radius=0.015, 
            color=[1, 1, 0]
        )
        geometries.append(function_sphere)
    
    # Add pre-contact waypoints (GREEN spheres)
    for waypoint in points_3d_dict['waypoints_3d']['pre_contact']:
        waypoint_sphere = create_sphere_at_point(waypoint, radius=0.012, color=[0, 1, 0])
        geometries.append(waypoint_sphere)
    
    # Add post-contact waypoints (CYAN spheres)
    for waypoint in points_3d_dict['waypoints_3d']['post_contact']:
        waypoint_sphere = create_sphere_at_point(waypoint, radius=0.012, color=[0, 1, 1])
        geometries.append(waypoint_sphere)
    
    # Draw lines connecting waypoints to target
    if points_3d_dict['keypoints_3d']['target'] is not None:
        target = points_3d_dict['keypoints_3d']['target']
        
        for waypoint in points_3d_dict['waypoints_3d']['pre_contact']:
            line = create_line_between_points(waypoint, target, color=[0, 1, 0])
            geometries.append(line)
        
        for waypoint in points_3d_dict['waypoints_3d']['post_contact']:
            line = create_line_between_points(target, waypoint, color=[0, 1, 1])
            geometries.append(line)
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Save point cloud if requested
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to: {save_path}")
    
    # Visualize
    print("\nVisualizing...")
    print("Legend: RED=Target, BLUE=Grasp, YELLOW=Function, GREEN=Pre-contact, CYAN=Post-contact")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Keypoints",
        width=1280,
        height=720
    )


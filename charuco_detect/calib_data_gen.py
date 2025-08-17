import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm
import os
import sys
import yaml

# Import the OpenCVRenderer from the same directory
from opencv_render import OpenCVRenderer

sys.setrecursionlimit(10000)

DEBUG = os.getenv('DEBUG', False)
AUGMENT = not DEBUG or os.getenv('AUGMENT', True)

# Image dimensions - using smaller size for faster generation
ORIG_HEIGHT = 1080
ORIG_WIDTH = 1920
SCALE_FACTOR = 2  # Reduce for faster processing

CURRENT_FILEPATH = Path(__file__).parent.absolute()

# ChArUco board configuration (from calibrate.py)
SQUARE_LENGTH = 500  # In world units (mm)
MARKER_LENGTH = 300
NUMBER_OF_SQUARES_VERTICALLY = 11
NUMBER_OF_SQUARES_HORIZONTALLY = 8

# Create ChArUco board
charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard(
    size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=charuco_marker_dictionary,
)

def create_background_textures():
    """Create simple background textures if they don't exist"""
    bg_dir = CURRENT_FILEPATH / "background_textures"
    bg_dir.mkdir(exist_ok=True)
    
    # Create some simple background patterns if they don't exist
    bg_files = list(bg_dir.glob('*.jpg'))
    if len(bg_files) == 0:
        print("Creating default background textures...")
        
        # Solid colors
        colors = [(50, 50, 50), (100, 100, 100), (150, 150, 150), (80, 60, 40)]
        for i, color in enumerate(colors):
            img = np.full((512, 512, 3), color, dtype=np.uint8)
            cv2.imwrite(str(bg_dir / f"solid_{i}.jpg"), img)
        
        # Gradient
        gradient = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            gradient[i, :] = [i//2, i//4, i//3]
        cv2.imwrite(str(bg_dir / "gradient.jpg"), gradient)
        
        # Noise pattern
        noise = np.random.randint(0, 100, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_dir / "noise.jpg"), noise)
        
        bg_files = list(bg_dir.glob('*.jpg'))
    
    return [cv2.imread(str(p)) for p in bg_files]

def generate_charuco_texture():
    """Generate ChArUco board texture"""
    # Generate board image
    board_size_pixels = (1080, 1920)  # Height, Width for generateImage
    margin_size = 50
    
    board_img = charuco_board.generateImage(board_size_pixels, marginSize=margin_size)
    # Convert to BGR for consistency
    board_img_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    
    return board_img_bgr

def sample_camera_pose_spherical(board_center: np.ndarray = np.array([0, 0, 0])):
    """
    Sample camera pose in spherical coordinates around the ChArUco board
    Biased toward closer distances for better corner detection
    """
    # Distance: 1cm to 1m, biased toward closer distances
    # Use exponential distribution to bias toward closer distances
    distance_min = 10.0   # 1cm in mm
    distance_max = 1000.0  # 1m in mm
    
    # Generate exponential distribution biased toward smaller values
    u = np.random.uniform(0, 1)
    # Inverse transform sampling for exponential-like distribution
    distance = distance_min + (distance_max - distance_min) * (1 - np.exp(-3 * u)) / (1 - np.exp(-3))
    
    # Spherical angles
    # Elevation: -60° to +60° (avoid extreme viewing angles)
    elevation = np.random.uniform(-60, 60)  # degrees
    # Azimuth: full 360°
    azimuth = np.random.uniform(0, 360)  # degrees
    
    # Convert spherical to Cartesian (camera position)
    elev_rad = np.radians(elevation)
    azim_rad = np.radians(azimuth)
    
    cam_x = distance * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = distance * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = distance * np.sin(elev_rad)
    
    camera_position = np.array([cam_x, cam_y, cam_z]) + board_center
    
    # Calculate look-at direction (camera points toward board center with some offset)
    look_at_offset = np.random.normal(0, 0.1, 3) * distance  # Small random offset
    look_at_target = board_center + look_at_offset
    
    # Calculate camera orientation using look-at
    forward = look_at_target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Up vector (approximately world up with some randomness)
    up_base = np.array([0, 0, 1])
    up_noise = np.random.normal(0, 0.2, 3)
    up = up_base + up_noise
    up = up / np.linalg.norm(up)
    
    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up to ensure orthogonality
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Create rotation matrix (camera to world)
    # Note: OpenCV uses different convention, so we need to be careful with directions
    rotation_matrix = np.column_stack([right, -up, -forward])  # Camera convention
    
    # Convert to rotation vector
    rvec = cv2.Rodrigues(rotation_matrix)[0].flatten()
    tvec = camera_position.flatten()
    
    return rvec.astype(np.float32), tvec.astype(np.float32)

def generate_board_pose():
    """Generate random pose for the ChArUco board"""
    # Random position in a reasonable workspace
    position = np.random.uniform(-200, 200, 3)  # ±20cm from origin
    position[2] = np.random.uniform(-100, 100)  # ±10cm in Z
    
    # Random orientation (avoid extreme rotations)
    max_rotation = 45  # degrees
    rotation_angles = np.random.uniform(-max_rotation, max_rotation, 3)
    
    # Convert to rotation vector
    rotation = Rotation.from_euler('xyz', rotation_angles, degrees=True)
    rvec = rotation.as_rotvec().astype(np.float32)
    tvec = position.astype(np.float32)
    
    return rvec, tvec

def generate_camera_matrices(num_cameras: int = 20):
    """Generate diverse camera matrices with wide fisheye characteristics"""
    camera_matrices = []
    distortion_coefficients_list = []
    
    for _ in range(num_cameras):
        # Base resolution
        width = ORIG_WIDTH // SCALE_FACTOR
        height = ORIG_HEIGHT // SCALE_FACTOR
        
        # Focal length range for wide fisheye lenses (shorter focal lengths)
        # For fisheye: focal length typically 0.3-0.8 times image width
        focal_length_base = np.random.uniform(0.3, 0.8) * width
        
        # Principal point near center with some variation
        cx = width / 2 + np.random.uniform(-width*0.1, width*0.1)
        cy = height / 2 + np.random.uniform(-height*0.1, height*0.1)
        
        # Slight asymmetry in focal lengths
        fx = focal_length_base * np.random.uniform(0.95, 1.05)
        fy = focal_length_base * np.random.uniform(0.95, 1.05)
        
        cam_mat = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Fisheye distortion coefficients (stronger distortion)
        # k1, k2, p1, p2, k3, k4, k5, k6
        k1 = np.random.uniform(-0.5, -0.1)  # Strong barrel distortion
        k2 = np.random.uniform(-0.3, 0.3)
        p1 = np.random.uniform(-0.01, 0.01)  # Tangential distortion
        p2 = np.random.uniform(-0.01, 0.01)
        k3 = np.random.uniform(-0.1, 0.1)
        
        # Additional coefficients for more complex distortion models
        k4 = np.random.uniform(-0.05, 0.05)
        k5 = np.random.uniform(-0.05, 0.05)
        k6 = np.random.uniform(-0.05, 0.05)
        
        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float32)
        
        camera_matrices.append(cam_mat)
        distortion_coefficients_list.append(dist_coeffs)
    
    return camera_matrices, distortion_coefficients_list

def extract_charuco_corners(charuco_texture, board_rvec, board_tvec, cam_rvec, cam_tvec, cam_matrix, dist_coeffs, img_shape):
    """Extract ChArUco corner positions for YOLO labeling"""
    
    # Get ChArUco board corners in board coordinate system
    charuco_corners_3d = []
    charuco_ids = []
    
    # ChArUco corners are at intersections of black and white squares
    # For an 8x11 board, we have (8-1)x(11-1) = 7x10 = 70 internal corners
    corner_id = 0
    for row in range(NUMBER_OF_SQUARES_VERTICALLY - 1):  # 10 rows of corners
        for col in range(NUMBER_OF_SQUARES_HORIZONTALLY - 1):  # 7 cols of corners
            # Corner position in board coordinates (mm)
            x = (col + 1) * SQUARE_LENGTH
            y = (row + 1) * SQUARE_LENGTH
            z = 0
            
            charuco_corners_3d.append([x, y, z])
            charuco_ids.append(corner_id)
            corner_id += 1
    
    charuco_corners_3d = np.array(charuco_corners_3d, dtype=np.float32)
    
    # Transform corners from board coordinates to world coordinates
    board_rotation = cv2.Rodrigues(board_rvec.reshape(3, 1))[0]
    corners_world = (board_rotation @ charuco_corners_3d.T + board_tvec.reshape(-1, 1)).T
    
    # Project to image coordinates
    corners_image = cv2.projectPoints(
        corners_world, cam_rvec.reshape(3, 1), cam_tvec.reshape(3, 1), cam_matrix, dist_coeffs
    )[0].squeeze()
    
    # Filter corners that are visible in the image
    valid_corners = []
    valid_ids = []
    
    height, width = img_shape[:2]
    
    for i, corner in enumerate(corners_image):
        x, y = corner
        if 0 <= x < width and 0 <= y < height:
            valid_corners.append([x, y])
            valid_ids.append(charuco_ids[i])
    
    return np.array(valid_corners), valid_ids

def create_yolo_labels(corners, corner_ids, img_shape):
    """Create YOLO format labels for ChArUco corners"""
    if len(corners) == 0:
        return ""
    
    height, width = img_shape[:2]
    
    # Create bounding box around all corners
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Add some padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    # YOLO bounding box format: center_x, center_y, width, height (normalized)
    bbox_center_x = (x_min + x_max) / 2 / width
    bbox_center_y = (y_min + y_max) / 2 / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height
    
    # Class 0 for ChArUco board
    label_parts = [f"0 {bbox_center_x:.6f} {bbox_center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"]
    
    # Add corner keypoints (normalized coordinates)
    for corner in corners:
        x_norm = corner[0] / width
        y_norm = corner[1] / height
        visibility = 2  # Visible
        label_parts.append(f"{x_norm:.6f} {y_norm:.6f} {visibility}")
    
    return " ".join(label_parts)

def augment_image(img: np.ndarray):
    """Apply augmentation effects (from synth_data.py)"""
    if not AUGMENT:
        return img
        
    img_copy = np.zeros_like(img)
    # add random shapes with 0-80% opacity
    n_shapes = np.random.randint(2, 10)
    
    for _ in range(n_shapes):
        shape = np.random.choice(['circle', 'rectangle'])
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        if shape == 'circle':
            center = np.random.uniform(0, 1, 2)
            radius = np.random.uniform(0.01, 0.3)
            cv2.circle(
                img_copy, 
                tuple((center * img.shape[:2]).astype(int)), 
                int(radius * img.shape[1]), 
                color, 
                -1
            )
        elif shape == 'rectangle':
            center = np.random.uniform(0, 1, 2)
            size = np.random.uniform(0.1, 0.3, 2)
            pt1 = ((center - size/2) * img.shape[:2]).astype(int)
            pt2 = ((center + size/2) * img.shape[:2]).astype(int)
            cv2.rectangle(img_copy, tuple(pt1), tuple(pt2), color, -1)
    
    # Add banding occasionally
    if np.random.uniform(0, 1) < 0.1:
        color = (204, 2, 187)
        start = 0
        while start < img_copy.shape[0]:
            gap = int(np.random.uniform(0.1, 0.7) * img_copy.shape[0])
            start += gap
            if start >= img_copy.shape[0]:
                break
            height = int(np.random.uniform(0.1, 0.6) * img_copy.shape[0])
            cv2.rectangle(img_copy, (0, start), (img_copy.shape[1]-1, start+height), color, -1)
            start += height
    
    # Blend with original
    opacity = np.random.uniform(0, 0.8)
    img = cv2.addWeighted(img, 1-opacity, img_copy, opacity, 0)
    
    # Blur
    kernel_size = np.random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    return img

def make_datapoint(charuco_texture, background_textures, camera_matrices, distortion_coefficients_list):
    """Generate a single training datapoint"""
    
    # Image dimensions
    height = ORIG_HEIGHT // SCALE_FACTOR
    width = ORIG_WIDTH // SCALE_FACTOR
    
    # Sample camera intrinsics
    cam_idx = np.random.randint(len(camera_matrices))
    cam_matrix = camera_matrices[cam_idx]
    dist_coeffs = distortion_coefficients_list[cam_idx]
    
    # Generate board pose
    board_rvec, board_tvec = generate_board_pose()
    
    # Generate camera pose (spherical around board)
    cam_rvec, cam_tvec = sample_camera_pose_spherical(board_tvec)
    
    # Create renderer
    renderer = OpenCVRenderer(cam_matrix, dist_coeffs)
    
    # Add background
    if background_textures:
        bg_texture = background_textures[np.random.randint(len(background_textures))]
        # Create large background plane
        bg_size = 5000  # 5m x 5m background
        bg_distance = np.random.uniform(2000, 5000)  # 2-5m behind
        bg_rvec = np.array([0, 0, 0], dtype=np.float32)
        bg_tvec = np.array([0, 0, -bg_distance], dtype=np.float32)
        renderer.add_billboard_from_pose_and_size(bg_texture, bg_rvec.reshape(3, 1), bg_tvec.reshape(3, 1), (bg_size, bg_size))
    
    # Add ChArUco board
    board_size_mm = (
        NUMBER_OF_SQUARES_HORIZONTALLY * SQUARE_LENGTH,
        NUMBER_OF_SQUARES_VERTICALLY * SQUARE_LENGTH
    )
    renderer.add_billboard_from_pose_and_size(charuco_texture, board_rvec.reshape(3, 1), board_tvec.reshape(3, 1), board_size_mm)
    
    # Render image (ensure proper shape for OpenCV)
    cam_rvec_shaped = cam_rvec.reshape(3, 1)
    cam_tvec_shaped = cam_tvec.reshape(3, 1)
    img = renderer.render_image((width, height), cam_rvec_shaped, cam_tvec_shaped)
    
    if img is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Extract ChArUco corners for labeling
    corners, corner_ids = extract_charuco_corners(
        charuco_texture, board_rvec, board_tvec, 
        cam_rvec, cam_tvec, cam_matrix, dist_coeffs, 
        (height, width)
    )
    
    # Create YOLO labels
    labels = create_yolo_labels(corners, corner_ids, (height, width))
    
    # Apply augmentation
    img = augment_image(img)
    
    if DEBUG and len(corners) > 0:
        # Visualize corners for debugging
        debug_img = img.copy()
        for corner in corners:
            cv2.circle(debug_img, tuple(corner.astype(int)), 3, (0, 255, 0), -1)
        cv2.imshow('debug', debug_img)
        cv2.waitKey(1)
    
    return img, labels

def create_dataset_yaml(dataset_path: Path, train_images_path: Path, val_images_path: Path):
    """Create Ultralytics YAML configuration file"""
    
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': str(train_images_path.absolute()),
        'val': str(val_images_path.absolute()),
        'nc': 1,  # Number of classes
        'names': ['charuco_board'],  # Class names
        'kpt_shape': [70, 3]  # 70 keypoints (ChArUco corners), 3 values each (x, y, visibility)
    }
    
    yaml_path = dataset_path / 'charuco_dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML created at: {yaml_path}")
    return yaml_path

def main():
    # Configuration
    DATASET_SIZE = 100 if DEBUG else 5000
    VAL_SPLIT = 0.1
    TRAIN_SIZE = int(DATASET_SIZE * (1 - VAL_SPLIT))
    VAL_SIZE = DATASET_SIZE - TRAIN_SIZE
    
    print(f"Generating {DATASET_SIZE} images ({TRAIN_SIZE} train, {VAL_SIZE} val)")
    
    # Setup directories
    dataset_dir = CURRENT_FILEPATH / 'charuco_dataset'
    dataset_dir.mkdir(exist_ok=True)
    
    imgs_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    imgs_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    train_images_dir = imgs_dir / 'train'
    train_labels_dir = labels_dir / 'train'
    val_images_dir = imgs_dir / 'val'
    val_labels_dir = labels_dir / 'val'
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Generate resources
    print("Generating ChArUco texture...")
    charuco_texture = generate_charuco_texture()
    
    print("Loading background textures...")
    background_textures = create_background_textures()
    
    print("Generating camera matrices...")
    camera_matrices, distortion_coefficients_list = generate_camera_matrices()
    
    if DEBUG:
        cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('debug', 800, 600)
    
    # Generate training data
    print("Generating training data...")
    for i in tqdm(range(TRAIN_SIZE)):
        img, labels = make_datapoint(charuco_texture, background_textures, camera_matrices, distortion_coefficients_list)
        
        cv2.imwrite(str(train_images_dir / f'{i:06d}.jpg'), img)
        
        if labels:  # Only write label file if there are valid labels
            with open(train_labels_dir / f'{i:06d}.txt', 'w') as f:
                f.write(labels)
    
    # Generate validation data
    print("Generating validation data...")
    for i in tqdm(range(VAL_SIZE)):
        img, labels = make_datapoint(charuco_texture, background_textures, camera_matrices, distortion_coefficients_list)
        
        cv2.imwrite(str(val_images_dir / f'{i:06d}.jpg'), img)
        
        if labels:  # Only write label file if there are valid labels
            with open(val_labels_dir / f'{i:06d}.txt', 'w') as f:
                f.write(labels)
    
    # Create Ultralytics YAML
    yaml_path = create_dataset_yaml(dataset_dir, train_images_dir, val_images_dir)
    
    print("\nDataset generation complete!")
    print(f"Dataset directory: {dataset_dir}")
    print(f"YAML config: {yaml_path}")
    print(f"Training images: {TRAIN_SIZE}")
    print(f"Validation images: {VAL_SIZE}")
    
    if DEBUG:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

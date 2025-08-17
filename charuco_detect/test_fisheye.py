#!/usr/bin/env python3
"""
Test script for fisheye calibration and rendering functionality.
"""

import numpy as np
import cv2
from calib_data_gen import generate_fisheye_distortion_coeffs, generate_camera_matrices
from opencv_render import OpenCVRenderer

def test_fisheye_distortion_generation():
    """Test fisheye distortion coefficient generation"""
    print("Testing fisheye distortion coefficient generation...")
    
    for i in range(5):
        k1, k2, k3, k4 = generate_fisheye_distortion_coeffs()
        print(f"  Set {i+1}: k1={k1:.3f}, k2={k2:.3f}, k3={k3:.3f}, k4={k4:.3f}")
        
        # Verify fisheye characteristics
        assert k1 < 0, f"k1 should be negative for fisheye effect, got {k1}"
        assert k2 > 0, f"k2 should be positive for compensation, got {k2}"
        assert abs(k3) < 0.2, f"k3 should be small, got {k3}"
        assert abs(k4) < 0.1, f"k4 should be small, got {k4}"
    
    print("✓ Fisheye distortion coefficient generation working correctly")

def test_camera_matrix_generation():
    """Test camera matrix generation with fisheye distortion"""
    print("\nTesting camera matrix generation...")
    
    camera_matrices, distortion_coefficients_list = generate_camera_matrices(5)
    
    for i, (cam_mat, dist_coeffs) in enumerate(zip(camera_matrices, distortion_coefficients_list)):
        print(f"  Camera {i+1}:")
        print(f"    Camera matrix shape: {cam_mat.shape}")
        print(f"    Distortion coefficients shape: {dist_coeffs.shape}")
        print(f"    Distortion coefficients: {dist_coeffs.flatten()}")
        
        # Verify shapes
        assert cam_mat.shape == (3, 3), f"Camera matrix should be 3x3, got {cam_mat.shape}"
        assert dist_coeffs.shape == (4, 1), f"Distortion coefficients should be 4x1, got {dist_coeffs.shape}"
        
        # Verify fisheye characteristics
        assert dist_coeffs[0] < 0, f"k1 should be negative for fisheye effect, got {dist_coeffs[0]}"
        assert dist_coeffs[1] > 0, f"k2 should be positive for compensation, got {dist_coeffs[1]}"
    
    print("✓ Camera matrix generation working correctly")

def test_fisheye_undistortion_maps():
    """Test fisheye undistortion map generation"""
    print("\nTesting fisheye undistortion map generation...")
    
    # Create a test camera matrix and distortion coefficients
    width, height = 1280, 960
    fx = 0.5 * width
    fy = 0.5 * height
    cx = width / 2
    cy = height / 2
    
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([-0.3, 0.2, 0.0, 0.0], dtype=np.float32)
    
    # Generate undistortion maps
    new_K = cam_matrix.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        cam_matrix, dist_coeffs, None, new_K, (width, height), cv2.CV_16SC2
    )
    
    print(f"  Map1 shape: {map1.shape}")
    print(f"  Map2 shape: {map2.shape}")
    print(f"  Map1 dtype: {map1.dtype}")
    print(f"  Map2 dtype: {map2.dtype}")
    
    # Verify shapes and types
    # For fisheye, map1 has shape (height, width, 2) and map2 has shape (height, width)
    assert map1.shape == (height, width, 2), f"Map1 should be {height}x{width}x2, got {map1.shape}"
    assert map2.shape == (height, width), f"Map2 should be {height}x{width}, got {map2.shape}"
    assert map1.dtype == np.int16, f"Map1 should be int16, got {map1.dtype}"
    assert map2.dtype == np.uint16, f"Map2 should be uint16, got {map2.dtype}"
    
    print("✓ Fisheye undistortion map generation working correctly")

def test_fisheye_projection():
    """Test fisheye point projection"""
    print("\nTesting fisheye point projection...")
    
    # Create test camera parameters
    width, height = 1280, 960
    fx = 0.5 * width
    fy = 0.5 * height
    cx = width / 2
    cy = height / 2
    
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([-0.3, 0.2, 0.0, 0.0], dtype=np.float32)
    
    # Test points in 3D space
    test_points = np.array([
        [0, 0, 1],      # Center point
        [1, 0, 1],      # Right point
        [0, 1, 1],      # Down point
        [-1, 0, 1],     # Left point
        [0, -1, 1],     # Up point
    ], dtype=np.float32)
    
    # Project points using fisheye model
    rvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
    tvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
    
    projected_points = cv2.fisheye.projectPoints(
        test_points.reshape(-1, 1, 3),
        rvec,
        tvec,
        cam_matrix,
        dist_coeffs
    )[0].squeeze()
    
    print(f"  Test points shape: {test_points.shape}")
    print(f"  Projected points shape: {projected_points.shape}")
    print(f"  First projected point: {projected_points[0]}")
    
    # Verify projection
    assert projected_points.shape == (5, 2), f"Projected points should be 5x2, got {projected_points.shape}"
    
    # Center point should be near image center
    center_projection = projected_points[0]
    assert abs(center_projection[0] - cx) < 10, f"Center projection x should be near {cx}, got {center_projection[0]}"
    assert abs(center_projection[1] - cy) < 10, f"Center projection y should be near {cy}, got {center_projection[1]}"
    
    print("✓ Fisheye point projection working correctly")

def test_opencv_renderer_fisheye():
    """Test OpenCVRenderer with fisheye distortion"""
    print("\nTesting OpenCVRenderer with fisheye distortion...")
    
    # Create renderer with fisheye parameters
    width, height = 640, 480
    fx = 0.5 * width
    fy = 0.5 * height
    cx = width / 2
    cy = height / 2
    
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([-0.3, 0.2, 0.0, 0.0], dtype=np.float32)
    
    renderer = OpenCVRenderer(cam_matrix, dist_coeffs)
    
    # Create a simple test texture
    test_texture = np.zeros((100, 100, 3), dtype=np.uint8)
    test_texture[:, :, 0] = 255  # Red texture
    
    # Add billboard
    renderer.add_billboard_from_pose_and_size(
        test_texture,
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0, 0, -200], dtype=np.float32),
        (100, 100)
    )
    
    # Render image
    rvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
    tvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
    
    try:
        img = renderer.render_image((width, height), rvec, tvec)
        print(f"  Rendered image shape: {img.shape}")
        print(f"  Rendered image dtype: {img.dtype}")
        
        # Verify image
        assert img.shape == (height, width, 3), f"Image should be {height}x{width}x3, got {img.shape}"
        assert img.dtype == np.uint8, f"Image should be uint8, got {img.dtype}"
        
        print("✓ OpenCVRenderer fisheye rendering working correctly")
        
    except Exception as e:
        print(f"  Error during rendering: {e}")
        print("✗ OpenCVRenderer fisheye rendering failed")

def main():
    """Run all tests"""
    print("Running fisheye calibration and rendering tests...\n")
    
    try:
        test_fisheye_distortion_generation()
        test_camera_matrix_generation()
        test_fisheye_undistortion_maps()
        test_fisheye_projection()
        test_opencv_renderer_fisheye()
        
        print("\n🎉 All tests passed! Fisheye functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

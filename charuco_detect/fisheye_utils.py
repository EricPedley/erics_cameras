#!/usr/bin/env python3
"""
Utility functions for fisheye camera operations.
"""

import numpy as np
import cv2

def create_fisheye_inverse_maps(cam_matrix, dist_coeffs, resolution):
    """
    Create inverse rectification maps for fisheye cameras.
    
    These maps go from undistorted coordinates back to distorted coordinates,
    which is what's needed for the rendering pipeline to work correctly.
    
    Args:
        cam_matrix (np.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (np.ndarray): 4x1 fisheye distortion coefficients [k1, k2, k3, k4]
        resolution (tuple): (width, height) of the image
        
    Returns:
        tuple: (map1, map2) where map1 contains x-coordinates and map2 contains y-coordinates
               Both maps are float32 arrays of shape (height, width)
    """
    alpha = 0
    height, width = resolution[1], resolution[0]
    
    # Create grid of undistorted coordinates
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Normalize coordinates
    a = (x_coords - cam_matrix[0, 2]) / cam_matrix[0, 0]
    b = (y_coords - cam_matrix[1, 2]) / cam_matrix[1, 1]
    
    # Convert to polar coordinates
    r = np.sqrt(a**2 + b**2)
    theta = np.arctan2(b, a)
    
    # Apply fisheye distortion model to get distorted radius
    # r_distorted = r * (1 + k1*r + k2*r² + k3*r³ + k4*r⁴)
    k1, k2, k3, k4 = dist_coeffs.flatten()
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    # theta_d = np.tan(theta)
    
    # Convert back to Cartesian coordinates
    x_prime = theta_d/r*a
    y_prime = theta_d/r*b
    
    # Convert back to pixel coordinates
    u = cam_matrix[0,0] * (x_prime + alpha*y_prime) + cam_matrix[0,2]
    v = cam_matrix[1,1] * y_prime + cam_matrix[1,2]
    
    # Create maps in the format expected by cv2.remap
    # For cv2.remap, we need separate x and y maps
    map1 = u.astype(np.float32)
    map2 = v.astype(np.float32)
    
    return map1, map2

def create_fisheye_undistortion_maps(cam_matrix, dist_coeffs, resolution):
    """
    Create forward undistortion maps for fisheye cameras.
    
    These maps go from distorted coordinates to undistorted coordinates.
    This is the standard fisheye undistortion operation.
    
    Args:
        cam_matrix (np.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (np.ndarray): 4x1 fisheye distortion coefficients [k1, k2, k3, k4]
        resolution (tuple): (width, height) of the image
        
    Returns:
        tuple: (map1, map2) where map1 contains x-coordinates and map2 contains y-coordinates
               Both maps are float32 arrays of shape (height, width)
    """
    new_K = cam_matrix.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        cam_matrix, dist_coeffs, None, new_K, resolution, cv2.CV_16SC2
    )
    
    # Convert to float32 format expected by cv2.remap
    map1 = map1.astype(np.float32)
    map2 = map2.astype(np.float32)
    
    return map1, map2

#!/usr/bin/env python3
"""
Demonstration script showing fisheye camera effects.
"""

import numpy as np
import cv2
from opencv_render import OpenCVRenderer
from calib_data_gen import generate_fisheye_distortion_coeffs

def create_checkerboard_texture(size=200, squares=8):
    """Create a checkerboard texture for visualization"""
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // squares
    
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                texture[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = [255, 255, 255]
            else:
                texture[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = [0, 0, 0]
    
    return texture

def create_grid_texture(size=400, spacing=20):
    """Create a grid texture for visualization"""
    texture = np.full((size, size, 3), 128, dtype=np.uint8)
    
    # Draw vertical lines
    for x in range(0, size, spacing):
        texture[:, x] = [255, 255, 255]
    
    # Draw horizontal lines
    for y in range(0, size, spacing):
        texture[y, :] = [255, 255, 255]
    
    # Draw center cross
    center = size // 2
    texture[center-2:center+3, :] = [255, 0, 0]  # Red horizontal
    texture[:, center-2:center+3] = [0, 0, 255]  # Blue vertical
    
    return texture

def demo_fisheye_effect():
    """Demonstrate fisheye camera effects with different distortion levels"""
    
    # Image dimensions
    width, height = 800, 600
    
    # Camera matrix (wide-angle fisheye)
    fx = 0.4 * width  # Short focal length for wide angle
    fy = 0.4 * height
    cx = width / 2
    cy = height / 2
    
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create textures
    checkerboard = create_checkerboard_texture(200, 8)
    grid = create_grid_texture(400, 20)
    
    # Different distortion levels to demonstrate
    distortion_levels = [
        (0.0, 0.0, 0.0, 0.0, "No distortion"),
        (-0.2, 0.1, 0.0, 0.0, "Mild fisheye"),
        (-0.4, 0.2, 0.0, 0.0, "Moderate fisheye"),
        (-0.6, 0.3, 0.0, 0.0, "Strong fisheye"),
        (-0.8, 0.4, 0.0, 0.0, "Very strong fisheye")
    ]
    
    print("Demonstrating fisheye camera effects...")
    print("=" * 50)
    
    for i, (k1, k2, k3, k4, description) in enumerate(distortion_levels):
        print(f"\n{description} (k1={k1}, k2={k2}, k3={k3}, k4={k4})")
        
        # Create distortion coefficients
        dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32).reshape(4, 1)
        
        # Create renderer
        renderer = OpenCVRenderer(cam_matrix, dist_coeffs)
        
        # Add textures at different positions
        renderer.add_billboard_from_pose_and_size(
            checkerboard,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, -300], dtype=np.float32),
            (200, 200)
        )
        
        renderer.add_billboard_from_pose_and_size(
            grid,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, -500], dtype=np.float32),
            (400, 400)
        )
        
        # Render image
        rvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        tvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        
        img = renderer.render_image((width, height), rvec, tvec)
        
        # Save image
        filename = f"fisheye_demo_{i:02d}_{description.replace(' ', '_').lower()}.png"
        cv2.imwrite(filename, img)
        print(f"  Saved: {filename}")
        
        # Show image
        cv2.imshow(f"Fisheye Effect: {description}", img)
        cv2.waitKey(1000)  # Show for 1 second
    
    cv2.destroyAllWindows()
    print(f"\nDemo complete! Generated {len(distortion_levels)} images with different fisheye effects.")

def demo_generated_distortion():
    """Demonstrate with randomly generated fisheye distortion coefficients"""
    
    print("\n" + "=" * 50)
    print("Demonstrating with randomly generated fisheye coefficients...")
    
    # Image dimensions
    width, height = 800, 600
    
    # Camera matrix
    fx = 0.4 * width
    fy = 0.4 * height
    cx = width / 2
    cy = height / 2
    
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create texture
    grid = create_grid_texture(400, 20)
    
    # Generate 3 random fisheye distortion sets
    for i in range(3):
        k1, k2, k3, k4 = generate_fisheye_distortion_coeffs()
        print(f"\nGenerated coefficients {i+1}: k1={k1:.3f}, k2={k2:.3f}, k3={k3:.3f}, k4={k4:.3f}")
        
        # Create distortion coefficients
        dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32).reshape(4, 1)
        
        # Create renderer
        renderer = OpenCVRenderer(cam_matrix, dist_coeffs)
        
        # Add grid texture
        renderer.add_billboard_from_pose_and_size(
            grid,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, -400], dtype=np.float32),
            (400, 400)
        )
        
        # Render image
        rvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        tvec = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        
        img = renderer.render_image((width, height), rvec, tvec)
        
        # Save image
        filename = f"fisheye_generated_{i+1:02d}.png"
        cv2.imwrite(filename, img)
        print(f"  Saved: {filename}")
        
        # Show image
        cv2.imshow(f"Generated Fisheye {i+1}", img)
        cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
    print(f"\nGenerated fisheye demo complete!")

def main():
    """Run the fisheye demonstration"""
    print("Fisheye Camera Effect Demonstration")
    print("=" * 50)
    
    try:
        # Demo with predefined distortion levels
        demo_fisheye_effect()
        
        # Demo with generated distortion coefficients
        demo_generated_distortion()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("Check the generated PNG files to see the fisheye effects.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

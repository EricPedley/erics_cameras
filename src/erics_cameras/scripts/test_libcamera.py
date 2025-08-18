#!/usr/bin/env python3
"""
Test script for LibCameraCam class to demonstrate exposure control functionality.
"""

import argparse
from erics_cameras import LibCameraCam
import time

def main():
    parser = argparse.ArgumentParser(description="Test LibCameraCam exposure control")
    parser.add_argument("--exposure_mode", choices=["auto", "manual"], default="auto",
                       help="Exposure mode")
    parser.add_argument("--exposure_time", type=int, default=10000,
                       help="Exposure time in microseconds (manual mode only)")
    parser.add_argument("--gain_mode", choices=["auto", "manual"], default="auto",
                       help="Gain mode")
    parser.add_argument("--analogue_gain", type=float, default=2.0,
                       help="Analogue gain (manual mode only)")
    parser.add_argument("--exposure_value", type=float, default=0.0,
                       help="Exposure value adjustment")
    parser.add_argument("--brightness", type=float, default=0.0,
                       help="Brightness adjustment (-1.0 to 1.0)")
    parser.add_argument("--contrast", type=float, default=0.0,
                       help="Contrast adjustment")
    parser.add_argument("--saturation", type=float, default=0.0,
                       help="Saturation adjustment")
    parser.add_argument("--sharpness", type=float, default=0.0,
                       help="Sharpness adjustment")
    
    args = parser.parse_args()
    
    print("Initializing LibCameraCam...")
    
    try:
        # Initialize camera with command line arguments
        camera = LibCameraCam(
            log_dir="./test_logs",
            resolution=LibCameraCam.ResolutionOption.R720P,
            camera_name="/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36",
            framerate=30,
            exposure_mode=LibCameraCam.ExposureMode(args.exposure_mode),
            exposure_time_us=args.exposure_time,
            gain_mode=LibCameraCam.GainMode(args.gain_mode),
            analogue_gain=args.analogue_gain,
            ae_enable=(args.exposure_mode == "auto"),
            exposure_value=args.exposure_value,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            sharpness=args.sharpness
        )
        
        print("Camera initialized successfully!")
        print("\nCurrent camera settings:")
        settings = camera.get_current_settings()
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\nTesting image capture...")
        for i in range(5):
            print(f"Capturing image {i+1}/5...")
            img = camera.take_image()
            if img is not None:
                print(f"  Image captured: {img.shape}")
            else:
                print("  Failed to capture image")
            time.sleep(1)
        
        print("\nTesting parameter changes...")
        
        # Test changing exposure mode
        if args.exposure_mode == "auto":
            print("Switching to manual exposure mode...")
            camera.set_exposure_mode(LibCameraCam.ExposureMode.MANUAL)
            camera.set_exposure_time(5000)  # 5ms
            print("  New settings applied")
        
        # Test changing gain
        print("Setting manual gain mode...")
        camera.set_gain_mode(LibCameraCam.GainMode.MANUAL)
        camera.set_analogue_gain(3.0)
        print("  New gain settings applied")
        
        # Test image quality adjustments
        print("Applying image quality adjustments...")
        camera.set_brightness(0.1)
        camera.set_contrast(1.1)
        camera.set_saturation(1.2)
        print("  Image quality settings applied")
        
        print("\nFinal camera settings:")
        settings = camera.get_current_settings()
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

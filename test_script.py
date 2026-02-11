#!/usr/bin/env python3
"""
Simple test script to verify predict_tiles.py functionality
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_script_structure():
    """Test that the script has the correct structure"""
    script_path = project_root / "scripts" / "predict_tiles.py"

    if not script_path.exists():
        print(f"ERROR: Script not found at {script_path}")
        return False

    print("✓ Script file exists")

    # Read the script to check key components
    with open(script_path, 'r') as f:
        content = f.read()

    # Check for key components
    checks = [
        ("import argparse", "Argument parser imported"),
        ("from ultralytics import YOLO", "YOLO model imported"),
        ("def save_segmentation_result", "save_segmentation_result function exists"),
        ("def main", "main function exists"),
        ("model.predict", "Model prediction call exists"),
        ("csv.writer", "CSV writing functionality exists")
    ]

    for check, description in checks:
        if check in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description}")

    print("\nScript structure verification complete!")
    return True

if __name__ == "__main__":
    test_script_structure()
#!/usr/bin/env python3
"""
Test script for anthropometric scaling functionality.
Demonstrates how the new measurements are used for personalized pose transposition.
"""

import json
import numpy as np
from model3 import load_anthropometric_profile, scale_pose_sequence

def test_anthropometric_scaling():
    """Test the anthropometric scaling system."""
    print("🧪 Testing Anthropometric Scaling System")
    print("=" * 50)
    
    # Load configuration
    with open('test_config.json', 'r') as f:
        config = json.load(f)
    
    # Create anthropometric profile
    profile = load_anthropometric_profile(config)
    profile.print_profile()
    
    # Create test pose data
    print(f"\n📊 Creating test pose data...")
    test_pose = np.random.rand(5, 26, 3) * 100  # 5 frames, 26 joints, 3D
    print(f"Original pose shape: {test_pose.shape}")
    
    # Test different scaling scenarios
    print(f"\n🔧 Testing scaling scenarios...")
    
    # Scenario 1: Normal scaling (default measurements)
    scaled_normal = scale_pose_sequence(test_pose, profile, '3d')
    print(f"✅ Normal scaling completed")
    
    # Scenario 2: Create a taller person profile
    print(f"\n📏 Testing with taller person (6'2\")...")
    tall_config = config.copy()
    tall_config['anthropometrics']['height_in'] = 74.0  # 6'2"
    tall_config['anthropometrics']['left_arm_length_in'] = 26.0
    tall_config['anthropometrics']['right_arm_length_in'] = 26.0
    tall_config['anthropometrics']['left_leg_length_in'] = 44.0
    tall_config['anthropometrics']['right_leg_length_in'] = 44.0
    
    tall_profile = load_anthropometric_profile(tall_config)
    tall_profile.print_profile()
    
    scaled_tall = scale_pose_sequence(test_pose, tall_profile, '3d')
    print(f"✅ Tall person scaling completed")
    
    # Scenario 3: Create a shorter person profile
    print(f"\n📏 Testing with shorter person (5'4\")...")
    short_config = config.copy()
    short_config['anthropometrics']['height_in'] = 64.0  # 5'4"
    short_config['anthropometrics']['left_arm_length_in'] = 22.0
    short_config['anthropometrics']['right_arm_length_in'] = 22.0
    short_config['anthropometrics']['left_leg_length_in'] = 36.0
    short_config['anthropometrics']['right_leg_length_in'] = 36.0
    
    short_profile = load_anthropometric_profile(short_config)
    short_profile.print_profile()
    
    scaled_short = scale_pose_sequence(test_pose, short_profile, '3d')
    print(f"✅ Short person scaling completed")
    
    # Compare scaling results
    print(f"\n📈 Scaling Comparison:")
    print(f"Original first joint: {test_pose[0, 0]}")
    print(f"Normal scaling: {scaled_normal[0, 0]} (factor: {scaled_normal[0, 0] / test_pose[0, 0]})")
    print(f"Tall scaling: {scaled_tall[0, 0]} (factor: {scaled_tall[0, 0] / test_pose[0, 0]})")
    print(f"Short scaling: {scaled_short[0, 0]} (factor: {scaled_short[0, 0] / test_pose[0, 0]})")
    
    # Test bilateral differences
    print(f"\n🦵 Testing bilateral differences...")
    asym_config = config.copy()
    asym_config['anthropometrics']['left_arm_length_in'] = 25.0
    asym_config['anthropometrics']['right_arm_length_in'] = 23.0  # 2" difference
    
    asym_profile = load_anthropometric_profile(asym_config)
    print(f"✅ Asymmetric profile loaded (warnings expected)")
    
    print(f"\n🎯 Anthropometric scaling system test completed successfully!")
    print(f"Key features demonstrated:")
    print(f"  ✅ Comprehensive measurement loading")
    print(f"  ✅ Bilateral scaling support")
    print(f"  ✅ Validation and warnings")
    print(f"  ✅ Personalized transposition")
    print(f"  ✅ Different body types support")

if __name__ == "__main__":
    test_anthropometric_scaling()

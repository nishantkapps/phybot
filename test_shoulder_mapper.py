#!/usr/bin/env python3
"""
Simple test script for shoulder mapper configuration.
Allows users to easily test different ROM settings and see results.
"""

import json
import numpy as np
from pathlib import Path
from model3 import (
    map_shoulders_to_dofs_3d_sequence,
    map_shoulders_to_dofs_2d_sequence,
    load_pose_sequence
)

def create_sample_config():
    """Create a sample configuration file for testing."""
    config = {
        "inputs": {
            "use_shoulder_mapper": True
        },
        "anthropometrics": {
            "shoulder_width_in": 15.5  # User measures their shoulder width
        },
        "joints": {
            "l_sh_idx": 6,   # LeftShoulder
            "l_el_idx": 8,   # LeftForeArm (elbow proxy)
            "r_sh_idx": 11,  # RightShoulder  
            "r_el_idx": 13   # RightForeArm (elbow proxy)
        },
        "rom_deg": {
            "shoulder_flexion": {
                "L": {"min": 0, "max": 140, "neutral": 5},
                "R": {"min": 0, "max": 150, "neutral": 0}
            },
            "shoulder_abduction": {
                "L": {"min": 0, "max": 130, "neutral": 3},
                "R": {"min": 0, "max": 150, "neutral": 0}
            }
        },
        "smoothing": {
            "alpha": 0.3
        },
        "debug": {
            "print_torso": True,
            "print_angles": True,
            "print_dof_demo": True
        }
    }
    
    with open("test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✅ Created test_config.json - edit this file to test your settings")
    return config

def test_shoulder_mapper(mode="3d", example="Ex1", idx=0, config_file="test_config.json"):
    """Test shoulder mapper with user configuration."""
    
    # Load user config
    if not Path(config_file).exists():
        print(f"❌ Config file {config_file} not found. Creating sample...")
        create_sample_config()
        print("📝 Please edit test_config.json with your measurements, then run again.")
        return
    
    with open(config_file, "r") as f:
        cfg = json.load(f)
    
    print(f"🔧 Testing shoulder mapper in {mode.upper()} mode")
    print(f"📁 Using config: {config_file}")
    
    # Load pose data
    try:
        pose_seq = load_pose_sequence(".", idx, example, mode, max_frames=100)
        print(f"📊 Loaded {pose_seq.shape[0]} frames from {example}")
    except Exception as e:
        print(f"❌ Failed to load pose data: {e}")
        return
    
    # Extract config values
    joints = cfg.get("joints", {})
    rom = cfg.get("rom_deg", {})
    anthro = cfg.get("anthropometrics", {})
    smoothing = cfg.get("smoothing", {})
    
    l_sh_idx = joints.get("l_sh_idx", 6)
    l_el_idx = joints.get("l_el_idx", 8) 
    r_sh_idx = joints.get("r_sh_idx", 11)
    r_el_idx = joints.get("r_el_idx", 13)
    
    shoulder_width_in = anthro.get("shoulder_width_in", 15.0)
    alpha = smoothing.get("alpha", 0.3)
    
    # ROM settings
    flex_rom = rom.get("shoulder_flexion", {})
    abd_rom = rom.get("shoulder_abduction", {})
    
    # Per-side ROM (support both formats)
    def get_rom_side(rom_dict, side):
        if side in rom_dict:
            return rom_dict[side]
        return rom_dict  # fallback to shared values
    
    l_flex = get_rom_side(flex_rom, "L")
    r_flex = get_rom_side(flex_rom, "R") 
    l_abd = get_rom_side(abd_rom, "L")
    r_abd = get_rom_side(abd_rom, "R")
    
    print(f"📏 Shoulder width: {shoulder_width_in} inches")
    print(f"🦾 L Flexion: {l_flex['min']}-{l_flex['max']}° (neutral: {l_flex['neutral']}°)")
    print(f"🦾 R Flexion: {r_flex['min']}-{r_flex['max']}° (neutral: {r_flex['neutral']}°)")
    print(f"🦾 L Abduction: {l_abd['min']}-{l_abd['max']}° (neutral: {l_abd['neutral']}°)")
    print(f"🦾 R Abduction: {r_abd['min']}-{r_abd['max']}° (neutral: {r_abd['neutral']}°)")
    
    # Test the mapper
    try:
        if mode == "3d":
            dofs = map_shoulders_to_dofs_3d_sequence(
                pose_seq,
                l_sh_idx=l_sh_idx, l_el_idx=l_el_idx,
                r_sh_idx=r_sh_idx, r_el_idx=r_el_idx,
                flex_rom_min_deg=min(l_flex["min"], r_flex["min"]),
                flex_rom_max_deg=max(l_flex["max"], r_flex["max"]),
                abd_rom_min_deg=min(l_abd["min"], r_abd["min"]),
                abd_rom_max_deg=max(l_abd["max"], r_abd["max"]),
                l_flex_neutral_deg=l_flex["neutral"],
                r_flex_neutral_deg=r_flex["neutral"],
                l_abd_neutral_deg=l_abd["neutral"],
                r_abd_neutral_deg=r_abd["neutral"],
                smoothing_alpha=alpha
            )
        else:  # 2d
            dofs = map_shoulders_to_dofs_2d_sequence(
                pose_seq,
                l_sh_idx=l_sh_idx, l_el_idx=l_el_idx,
                r_sh_idx=r_sh_idx, r_el_idx=r_el_idx,
                shoulder_width_in=shoulder_width_in,
                flex_rom_min_deg=min(l_flex["min"], r_flex["min"]),
                flex_rom_max_deg=max(l_flex["max"], r_flex["max"]),
                abd_rom_min_deg=min(l_abd["min"], r_abd["min"]),
                abd_rom_max_deg=max(l_abd["max"], r_abd["max"]),
                l_flex_neutral_deg=l_flex["neutral"],
                r_flex_neutral_deg=r_flex["neutral"],
                l_abd_neutral_deg=l_abd["neutral"],
                r_abd_neutral_deg=r_abd["neutral"],
                smoothing_alpha=alpha
            )
        
        print(f"✅ Generated {dofs.shape[0]} DOF frames: [L_flex, R_flex, L_abd, R_abd]")
        
        # Show statistics
        print("\n📈 DOF Statistics:")
        for i, name in enumerate(["L_flex", "R_flex", "L_abd", "R_abd"]):
            col = dofs[:, i]
            print(f"  {name}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")
        
        # Show first few frames
        print(f"\n🔍 First 5 frames:")
        for t in range(min(5, dofs.shape[0])):
            print(f"  Frame {t}: L_flex={dofs[t,0]:.3f}, R_flex={dofs[t,1]:.3f}, L_abd={dofs[t,2]:.3f}, R_abd={dofs[t,3]:.3f}")
            
    except Exception as e:
        print(f"❌ Mapper failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Interactive test interface."""
    print("🧪 Shoulder Mapper Test Tool")
    print("=" * 40)
    
    # Get user preferences
    mode = input("Mode (2d/3d) [3d]: ").strip() or "3d"
    example = input("Example (Ex1/Ex2/...) [Ex1]: ").strip() or "Ex1"
    idx = int(input("Sequence index [0]: ").strip() or "0")
    config_file = input("Config file [test_config.json]: ").strip() or "test_config.json"
    
    print()
    test_shoulder_mapper(mode, example, idx, config_file)

if __name__ == "__main__":
    main()


# robotic_config.py

# =======================================================================
# 1. ORIGINAL BENCHMARK CONFIG (For embodied AI model alignment)
# =======================================================================
ORIGINAL_CONFIG = {
    # Physics & Controller limits
    "MAX_POS_STEP": 0.05,        # 5cm max movement per step at 20Hz (1.0 m/s cap)
    "MAX_REACH": 0.855,          # Panda wrist workspace radius, measured from robot0_base
    "KP_STIFFNESS": 150.0,
    "OSC_POS_ACTION_SCALE": 1.0,
    "OSC_ROT_OUTPUT_MAX": 0.5,
    
    # Waypoint Z-Offsets
    "APPROACH_Z": 0.05,
    "GRASP_Z_OFFSET": -0.012,
    "PLACE_Z_OFFSET": 0.05,
    "RETRACT_Z": 0.08,
    
    # Tolerances (Rigid stop-and-go)
    "HOVER_TOLERANCE": 0.08,     # Must come to a near stop above object
    "DIVE_TOLERANCE": 0.015,     # Must come to a complete stop before gripping
    "YAW_TOLERANCE": 0.04,
    
    # Timers
    "GRASP_WAIT_STEPS": 10   ,    # 0.5 seconds waiting for fingers to close
    "GRASP_YAW_OFFSET_DEG": 0.0,
    
}

# =======================================================================
# 2. INDUSTRY TAKT CONFIG (Optimized for 2-3s cycle times)
# =======================================================================
INDUSTRY_CONFIG = {
    # Physics & Controller limits
    "MAX_POS_STEP": 0.15,        # 15cm max movement per step (3.0 m/s cap)
    "MAX_REACH": 0.855,          # Panda wrist workspace radius, measured from robot0_base
    "KP_STIFFNESS": 250.0,       # Higher stiffness to track fast trajectories
    "OSC_POS_ACTION_SCALE": 1.0,
    "OSC_ROT_OUTPUT_MAX": 1.0,   # Faster wrist rotation
    
    # Waypoint Z-Offsets
    "APPROACH_Z": 0.10,
    "GRASP_Z_OFFSET": -0.012,
    "PLACE_Z_OFFSET": 0.05,
    "RETRACT_Z": 0.08,
    
    # Tolerances (Parabolic / Sweeping trajectories)
    "HOVER_TOLERANCE": 0.15,     # Trigger dive while still flying horizontally
    "DIVE_TOLERANCE": 0.03,      # Grip as soon as the bounding box is entered
    "YAW_TOLERANCE": 0.08,
    
    # Timers
    "GRASP_WAIT_STEPS": 8,     
    "GRASP_YAW_OFFSET_DEG": 0.0
}

# Toggle this variable to switch modes globally
ACTIVE_CONFIG = ORIGINAL_CONFIG

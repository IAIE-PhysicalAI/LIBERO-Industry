import os
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

# ==========================================
# SAFE CONFIGURATION VARIABLES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = os.path.join(BASE_DIR, "print_scaling_report.xlsx")

# --- INSERT YOUR 6 OBJECT NAMES HERE ---
# Exact folder/object names you are using for real-world validation
SELECTED_OBJECTS_FOR_VALIDATION = [
    "mario", 
    "mini_roller",
    "thomas",
    "porter",
    "mini_fire_engine",
    "school_bus",
    "conveyor_belt"
]

# Gripper Configuration (Adjust these values)
SIM_GRIPPER_OPEN_M = 0.079      # e.g., Panda gripper max opening in simulation (80mm)
REAL_GRIPPER_OPEN_MM = 70.0    # Real-world physical gripper opening in mm

# Strict precision threshold for reporting (mm)
PRECISION_TOLERANCE_MM = 0.1 

VARIANTS_TO_CHECK = ["", "_upright", "_flat", "_flat_x", "_flat_y"]
# ==========================================

def get_mesh_bounds(obj_path):
    """Calculates the raw unscaled bounding box of an OBJ file."""
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    valid = False
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                valid = True
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
                
    if not valid:
        return None
    return (max_x - min_x, max_y - min_y, max_z - min_z)

def analyze_object_for_printing(obj_name, variant, obj_dir):
    xml_filename = f"{obj_name}{variant}.xml"
    xml_path = os.path.join(obj_dir, xml_filename)
    if not os.path.exists(xml_path): 
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError: 
        return None

    # Identify the primary mesh and its XML scale
    obj_path = None
    xml_scale = (1.0, 1.0, 1.0)
    
    for geom in root.findall(".//geom"):
        if geom.get("class") == "collision":
            continue
            
        mesh_name = geom.get("mesh")
        asset_mesh = root.find(f".//asset/mesh[@name='{mesh_name}']")
        
        if asset_mesh is not None and asset_mesh.get("file"):
            obj_path = os.path.join(obj_dir, asset_mesh.get("file"))
            scale_str = asset_mesh.get("scale", "1 1 1")
            xml_scale = tuple(float(s) for s in scale_str.split())
            break 

    if not obj_path or not os.path.exists(obj_path):
        return None

    raw_bounds = get_mesh_bounds(obj_path)
    if not raw_bounds:
        return None

    # Calculate Simulation Size (Meters)
    sim_size_x = raw_bounds[0] * xml_scale[0]
    sim_size_y = raw_bounds[1] * xml_scale[1]
    sim_size_z = raw_bounds[2] * xml_scale[2]

    # Typically, a parallel jaw gripper approaches from the top and grasps the shortest planar axis
    sim_grasp_dim = min(sim_size_x, sim_size_y)
    
    # Calculate Real World Target Size (mm)
    sim_gripper_mm = SIM_GRIPPER_OPEN_M * 1000.0
    scale_ratio = REAL_GRIPPER_OPEN_MM / sim_gripper_mm
    
    target_real_x_mm = (sim_size_x * 1000.0) * scale_ratio
    target_real_y_mm = (sim_size_y * 1000.0) * scale_ratio
    target_real_z_mm = (sim_size_z * 1000.0) * scale_ratio

    # Calculate the scale factor needed for the 3D Printer Slicer
    # Slicer Scale = XML Scale * (Real Gripper / Sim Gripper)
    slicer_scale_x = xml_scale[0] * scale_ratio
    slicer_scale_y = xml_scale[1] * scale_ratio
    slicer_scale_z = xml_scale[2] * scale_ratio

    return {
        "Object": obj_name,
        "Variant": xml_filename,
        "Sim_Grasp_Width_m": round(sim_grasp_dim, 4),
        "Target_Real_X_mm": round(target_real_x_mm, 1),
        "Target_Real_Y_mm": round(target_real_y_mm, 1),
        "Target_Real_Z_mm": round(target_real_z_mm, 1),
        "Slicer_Scale_X": round(slicer_scale_x, 4),
        "Slicer_Scale_Y": round(slicer_scale_y, 4),
        "Slicer_Scale_Z": round(slicer_scale_z, 4),
        "Precision_Note": f"±{PRECISION_TOLERANCE_MM}mm"
    }

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory not found: {BASE_DIR}")
        return

    objects = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    objects.sort()

    results = []
    for obj in objects:
        for var in VARIANTS_TO_CHECK:
            data = analyze_object_for_printing(obj, var, os.path.join(BASE_DIR, obj))
            if data:
                results.append(data)

    if results:
        df = pd.DataFrame(results)
        
        # Filter for the target objects
        validation_df = df[df["Object"].isin(SELECTED_OBJECTS_FOR_VALIDATION)]
        
        # Write to multiple sheets
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            if not validation_df.empty:
                validation_df.to_excel(writer, sheet_name="Selected Validation Objects", index=False)
            df.to_excel(writer, sheet_name="All Objects", index=False)
            
        print(f"Extraction complete. 3D Printing requirements saved to {OUTPUT_FILE}")
        
        # --- PRINT GRIPPER CONFIGURATION ---
        print("\n--- Gripper Configuration ---")
        print(f"Simulation Gripper Max Opening: {SIM_GRIPPER_OPEN_M * 1000.0:.1f} mm")
        print(f"Real-World Gripper Max Opening: {REAL_GRIPPER_OPEN_MM:.1f} mm")
        scale_ratio = REAL_GRIPPER_OPEN_MM / (SIM_GRIPPER_OPEN_M * 1000.0)
        print(f"Derived Scale Multiplier:       {scale_ratio:.4f}")
        
        if not validation_df.empty:
            print("\n--- Slicer Scale Preview (Selected Objects) ---")
            for _, row in validation_df.iterrows():
                print(f"{row['Object']}{row['Variant']}: Scale XYZ -> ({row['Slicer_Scale_X']}, {row['Slicer_Scale_Y']}, {row['Slicer_Scale_Z']})")
        else:
            print("\nWarning: None of the names in SELECTED_OBJECTS_FOR_VALIDATION matched the parsed objects.")
            
    else:
        print("No valid objects processed.")

if __name__ == "__main__":
    main()

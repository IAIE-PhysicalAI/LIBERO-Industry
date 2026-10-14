"""
This script converts the xml file of inlays into robosuite-mjcf format. 
1. run obj2mjcf to cut up concave objects into collision meshes 
2. run this script to convert format 

"""

import os
import re
import glob
import math
import shutil
from pathlib import Path

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# If True, the script will automatically find ALL folders ending with "_inlay"
AUTO_FIND_INLAYS = False

# Specific targets if AUTO_FIND_INLAYS is False
TARGET_OBJECTS = [
    "mario_flat_inlay",
    "mini_fire_engine_flat_inlay",
    "mini_roller_flat_inlay",
    "porter_flat_inlay",
    "thomas_flat_inlay",
    "school_bus_flat_inlay"
]

CLEANUP_OLD_XML = True 

INDUSTRY_OBJECTS_PATH = BASE_DIR.parent.parent / "envs" / "objects" / "industry_objects.py"
# ==========================================

def calculate_sites_from_obj(obj_path):
    """
    Parses the visual OBJ to find the outer block dimensions.
    """
    if not os.path.exists(obj_path):
        return -0.01, 0.01, "0.05000 0 0"
    
    min_z = float('inf')
    max_z = float('-inf')
    max_r_sq = -1.0
    best_x, best_y = 0.0, 0.0
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    min_z = min(min_z, z)
                    max_z = max(max_z, z)
                    
                    r_sq = x**2 + y**2
                    if r_sq > max_r_sq:
                        max_r_sq = r_sq
                        best_x, best_y = x, y
                        
    if min_z == float('inf'):
        return -0.01, 0.01, "0.05000 0 0"
        
    radius_pos = f"{best_x:.5f} {best_y:.5f} 0"
    return min_z, max_z, radius_pos

def process_inlay(obj_name, base_dir):
    print(f"\n--- Processing Inlay: {obj_name} ---")
    obj_dir = os.path.join(base_dir, obj_name)
    
    if not os.path.exists(obj_dir):
        print(f"  [Error] Directory not found: {obj_dir}")
        return

    # 1. Flatten the obj2mjcf Nested Directory
    print("  [Step 1] Flattening obj2mjcf subfolders...")
    nested_dir = os.path.join(obj_dir, obj_name)
    if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
        # Move all collision files up one level
        for filename in os.listdir(nested_dir):
            if filename.endswith(".obj"):
                src = os.path.join(nested_dir, filename)
                dst = os.path.join(obj_dir, filename)
                # Overwrite if exists to be safe
                if os.path.exists(dst): os.remove(dst)
                shutil.move(src, dst)
                print(f"    Moved: {filename} to parent directory.")
        
        # Remove the now-empty subfolder
        try:
            os.rmdir(nested_dir)
            print("    Removed empty nested directory.")
        except OSError:
            print("    [Warning] Nested directory not empty, left intact.")

    # 2. Rename Visual Mesh to model.obj
    visual_candidates = [f"{obj_name}.obj", "model.obj"]
    for cand in visual_candidates:
        cand_path = os.path.join(obj_dir, cand)
        if os.path.exists(cand_path) and cand != "model.obj":
            os.rename(cand_path, os.path.join(obj_dir, "model.obj"))
            print(f"    Renamed visual mesh: {cand} -> model.obj")
            break

    # 3. Organize Collision Meshes
    print("  [Step 2] Organizing collision meshes...")
    search_pattern = os.path.join(obj_dir, "*_collision_*.obj")
    col_files = glob.glob(search_pattern)
    
    if not col_files:
        print("    [Warning] No convex collision meshes found! Visual mesh will be used.")
        num_collisions = 0
    else:
        def extract_num(filepath):
            match = re.search(r'collision_(\d+)\.obj', filepath)
            return int(match.group(1)) if match else -1

        col_files.sort(key=extract_num)
        
        # Safe sequential renaming
        temp_files = []
        for i, file_path in enumerate(col_files):
            temp_path = os.path.join(obj_dir, f"temp_col_{i}.obj")
            os.rename(file_path, temp_path)
            temp_files.append(temp_path)
            
        for i, temp_path in enumerate(temp_files):
            final_path = os.path.join(obj_dir, f"model_collision_{i}.obj")
            os.rename(temp_path, final_path)
            
        num_collisions = len(temp_files)
        print(f"    Renumbered {num_collisions} collision meshes sequentially.")

    # 4. Calculate Bounds
    print("  [Step 3] Calculating physical bounds...")
    model_obj_path = os.path.join(obj_dir, "model.obj")
    min_z, max_z, radius_pos = calculate_sites_from_obj(model_obj_path)

    # 5. Generate the XML
    print("  [Step 4] Generating flattened XML with Goal Region...")
    asset_meshes = ""
    collision_geoms = ""
    
    for i in range(num_collisions):
        asset_meshes += f'    <mesh name="model_collision_{i}" file="model_collision_{i}.obj"/>\n'
        collision_geoms += f'        <geom mesh="model_collision_{i}" class="collision"/>\n'

    if num_collisions == 0:
        collision_geoms = '        <geom mesh="model" class="collision"/>\n'

    xml_output_path = os.path.join(obj_dir, f"{obj_name}.xml")
    
    xml_content = f"""<mujoco model="{obj_name}">
  <default>
    <default class="visual">
      <geom group="1" type="mesh" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
    </default>
    <default class="collision">
    <geom group="0" type="mesh" friction="1.0 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </default>
  </default>
  
  <asset>
    <mesh name="model" file="model.obj"/>
{asset_meshes.rstrip()}
  </asset>
  
  <worldbody>
    <body>
      <body name="object" pos="0 0 0" quat="1 0 0 0">
        <inertial pos="0 0 0" mass="0.8" diaginertia="1e-3 1e-3 1e-3" />
        <geom mesh="model" class="visual"/>
{collision_geoms.rstrip()}
      <site name="goal_region" type="box" size="0.01 0.01 0.002" pos="0 0 0.005" rgba="0 1 0 0.3" quat="1 0 0 0" group="0"/>

      </body>
      
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {min_z:.5f}" quat="1 0 0 0" name="bottom_site"/>        
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {max_z:.5f}" quat="1 0 0 0" name="top_site"/>   
      <site rgba="0 0 0 0" size="0.01" pos="{radius_pos}" quat="1 0 0 0" name="horizontal_radius_site"/>
      
      
    </body>
  </worldbody>
</mujoco>
"""

    with open(xml_output_path, 'w') as f:
        f.write(xml_content)
    print(f"    Generated flat XML: {os.path.basename(xml_output_path)}")

    # 6. Cleanup
    if CLEANUP_OLD_XML:
        # Delete any leftover obj2mjcf xmls that don't match our naming standard
        for bad_xml in glob.glob(os.path.join(obj_dir, "*.xml")):
            if os.path.basename(bad_xml) != f"{obj_name}.xml":
                os.remove(bad_xml)
                print(f"    Cleaned up old XML: {os.path.basename(bad_xml)}")

def snake_to_camel(snake_str):
    """Converts snake_case to CamelCase for Python class names."""
    return "".join(x.title() for x in snake_str.split('_'))

def auto_register_object(obj_name, filepath):
    """
    Checks if the object is already in the allow/register list in industry_objects.py.
    If not, appends the boilerplate class registration to the end of the file.
    """
    if not os.path.exists(filepath):
        print(f"  [Error] Registration file not found: {filepath}")
        return

    # Convert "mouse_naga_upright_inlay" to "MouseNagaUprightInlay"
    class_name = snake_to_camel(obj_name)
    
    # Read the existing file to check for the class
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Check if the class is already registered
    if f"class {class_name}(IndustryObject):" not in content:
        print(f"  [Auto-Register] Adding new class {class_name} to industry_objects.py")
        
        # Create the standard boilerplate code matching your other objects
        registration_code = f"""
@register_object
class {class_name}(IndustryObject):
    def __init__(
        self,
        name="{obj_name}",
        obj_name="{obj_name}",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="{obj_name}", joints=joints)
        self.rotation = {{"x": (0, 0), "y": (0, 0), "z": (0, 0)}}
        self.rotation_axis = None
        self.z_offset = 0.0100
"""
        # Append the new class to the bottom of the file
        with open(filepath, 'a') as f:
            f.write(registration_code)
    else:
        print(f"  [Auto-Register] {class_name} is already registered. Skipping.")



if __name__ == "__main__":
    print("Starting Inlay Processing Pipeline...")
    
    if AUTO_FIND_INLAYS: 
        targets = [d for d in os.listdir(BASE_DIR) 
                  if os.path.isdir(os.path.join(BASE_DIR, d)) and d.endswith("_inlay")]
    else:
        targets = TARGET_OBJECTS 
        
    for obj in targets: 
        process_inlay(obj, BASE_DIR) 
        
        # --- NEW: Call the auto-registration check ---
        auto_register_object(obj, INDUSTRY_OBJECTS_PATH)
        
    print("\n--- Pipeline Complete! ---")

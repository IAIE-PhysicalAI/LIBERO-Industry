"""
This script cleans up and standardizes manually edited MuJoCo scanned objects 
(e.g., removing unwanted scanned cables or messy geometry) before pose generation.
"""
import os
import re
import glob
import math
from pathlib import Path

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_FILE = BASE_DIR.parent.parent / "envs" / "objects" / "industry_objects.py"

# Add the folder names of all objects you want to process
TARGET_OBJECTS = [
    "cream_bottle",
    "cream_jar",
    "serum_bottle",


    # 'thomas',
    # 'porter',
    # 'mario',
    # 'mini_fire_engine',
    # 'mini_roller',
    # 'school_bus',
    # 'grimlock',
    # 'dragon',
    # 'bumblebee',
]

# If True, deletes any existing "model.xml" so Robosuite/LIBERO doesn't get confused
# by having both model.xml and {object_name}.xml in the same folder.
CLEANUP_OLD_XML = True 
# ==========================================

def calculate_sites_from_objs(obj_paths, rotation_type='upright'):
    """
    Parses OBJ files across visual and collision meshes, applying 
    rotations to correctly locate bounds and horizontal radius.
    """
    min_z = float('inf')
    max_z = float('-inf')
    max_r_sq = -1.0
    best_x, best_y = 0.0, 0.0
    
    for obj_path in obj_paths:
        if not os.path.exists(obj_path): continue
        
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # Apply XML rotations to coordinates prior to calculation
                        if rotation_type == 'x_roll':
                            nx, ny, nz = x, -z, y
                        elif rotation_type == 'y_pitch':
                            nx, ny, nz = z, y, -x
                        else:
                            nx, ny, nz = x, y, z
                        
                        min_z = min(min_z, nz)
                        max_z = max(max_z, nz)
                        
                        r_sq = nx**2 + ny**2
                        if r_sq > max_r_sq:
                            max_r_sq = r_sq
                            best_x = nx
                            best_y = ny
                            
    if min_z == float('inf'):
        print(f"  [Warning] Missing OBJ data. Using default site values.")
        return -0.01, 0.01, "0.05000 0 0"
        
    radius_pos = f"{best_x:.5f} {best_y:.5f} 0"
    return min_z, max_z, radius_pos


def process_object(obj_name, base_dir, registry_path):
    print(f"\n--- Processing {obj_name} ---")
    obj_dir = os.path.join(base_dir, obj_name)
    
    if not os.path.exists(obj_dir):
        print(f"  [Error] Directory not found: {obj_dir}")
        return

    # 1. File Cleanup
    print("  [Step 1] Cleaning up messy filenames and .mtl files...")
    for filename in os.listdir(obj_dir):
        
        # Identify main visual mesh exported with numeric suffixes (e.g., model_001.obj, model_008.obj)
        # ^model_\d+\.obj$ cleanly separates the visual mesh from model_collision_*.obj
        if re.match(r'^model_\d+\.obj$', filename):
            old_path = os.path.join(obj_dir, filename)
            new_path = os.path.join(obj_dir, "model.obj")
            os.rename(old_path, new_path)
            print(f"    Renamed visual mesh: {filename} -> model.obj")
            
        # Catch-all for other files with "_001" suffix (like textures), strictly avoiding collisions
        elif "_001" in filename and "collision" not in filename:
            old_path = os.path.join(obj_dir, filename)
            new_path = os.path.join(obj_dir, filename.replace("_001", ""))
            os.rename(old_path, new_path)
            print(f"    Renamed artifact: {filename} -> {os.path.basename(new_path)}")
            
    # Remove .mtl files
    mtl_files = glob.glob(os.path.join(obj_dir, "*.mtl"))
    for mtl_file in mtl_files:
        os.remove(mtl_file)
        print(f"    Removed: {os.path.basename(mtl_file)}")

    # 2. Sequential Renaming of Collision Files
    print("  [Step 2] Organizing collision meshes...")
    search_pattern = os.path.join(obj_dir, "model_collision_*.obj")
    col_files = glob.glob(search_pattern)
    
    if not col_files:
        print("    No explicit collision files found. Visual mesh will be used for collision.")
        num_collisions = 0
    else:
        def extract_num(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'model_collision_(\d+)\.obj', filename)
            return int(match.group(1)) if match else -1

        col_files.sort(key=extract_num)
        
        # Rename to temp files first to avoid accidental overwriting
        temp_files = []
        for i, file_path in enumerate(col_files):
            temp_path = os.path.join(obj_dir, f"temp_col_{i}.obj")
            os.rename(file_path, temp_path)
            temp_files.append(temp_path)
            
        # Rename to final target names
        for i, temp_path in enumerate(temp_files):
            final_path = os.path.join(obj_dir, f"model_collision_{i}.obj")
            os.rename(temp_path, final_path)
            
        num_collisions = len(temp_files)
        print(f"    Sequentially renumbered {num_collisions} collision meshes.")

    # 3. Calculate Sites dynamically
    print("  [Step 3] Calculating physical bounds from all obj meshes...")
    
    # Gather model.obj + all collision objs
    obj_paths = [os.path.join(obj_dir, "model.obj")]
    for i in range(num_collisions):
        obj_paths.append(os.path.join(obj_dir, f"model_collision_{i}.obj"))
        
    # Deduce rotation type from filename if needed (matches XML output behavior)
    rotation_type = 'upright'
    if "_flat_x" in obj_name: rotation_type = 'x_roll'
    elif "_flat_y" in obj_name: rotation_type = 'y_pitch'
    
    min_z, max_z, radius_pos = calculate_sites_from_objs(obj_paths, rotation_type)
    print(f"    Bounds -> Bottom Z: {min_z:.4f}, Top Z: {max_z:.4f}, Radius Pos: {radius_pos}")


    # 4. Generate the XML
    print("  [Step 4] Generating flattened XML...")
    
    # --- NEW: Check for texture file ---
    if os.path.exists(os.path.join(obj_dir, "texture.png")):
        material_assets = '    <texture type="2d" name="texture" file="texture.png"/>\n    <material name="material_0" texture="texture" specular="0.5" shininess="0.5"/>'
        print("    Found texture.png. Adding texture map to XML.")
    else:
        material_assets = '    <material name="material_0" specular="0.5" shininess="0.5" rgba="0.8 0.8 0.8 1"/>'
        print("    No texture.png found. Generating default solid material.")
    # -----------------------------------

    asset_meshes = ""
    collision_geoms = ""
    
    for i in range(num_collisions):
        asset_meshes += f'    <mesh name="model_collision_{i}" file="model_collision_{i}.obj"/>\n'
        collision_geoms += f'        <geom mesh="model_collision_{i}" class="collision"/>\n'

    # Fallback: if no convex decomposition meshes exist, just use the main model for collision
    if num_collisions == 0:
        collision_geoms = '        <geom mesh="model" class="collision"/>\n'

    # Target XML name based on the folder (e.g., shoe_left.xml)
    xml_output_path = os.path.join(obj_dir, f"{obj_name}.xml")
    
    # Flat structure template: geometry sits DIRECTLY under body name="object"
    xml_content = f"""<mujoco model="{obj_name}">
  <default>
    <default class="visual">
      <geom group="1" type="mesh" contype="0" conaffinity="0"/>
    </default>
    <default class="collision">
      <geom group="0" type="mesh"/>
    </default>
  </default>
  <asset>
{material_assets}
    <mesh name="model" file="model.obj"/>
{asset_meshes.rstrip()}
  </asset>
  
  <worldbody>
    <body>
      <body name="object" pos="0 0 0" quat="1 0 0 0">
        <inertial pos="0 0 0" mass="0.2" diaginertia="1e-3 1e-3 1e-3" />
        <geom material="material_0" mesh="model" class="visual"/>
{collision_geoms.rstrip()}
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
    print(f"    Generated flat XML at: {os.path.basename(xml_output_path)}")

    # 5. Clean up old model.xml to prevent ambiguity
    print("  [Step 5] Checking for old model.xml files...")
    if CLEANUP_OLD_XML and xml_output_path != os.path.join(obj_dir, "model.xml"):
        old_model_xml = os.path.join(obj_dir, "model.xml")
        if os.path.exists(old_model_xml):
            os.remove(old_model_xml)
            print("    Removed old 'model.xml' to avoid duplicates.")

if __name__ == "__main__":
    print("Starting unified object processing...")
    for obj in TARGET_OBJECTS:
        process_object(obj, BASE_DIR, REGISTRY_FILE)
    print("\n--- Script complete! ---")

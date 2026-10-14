import os
import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_FILE = BASE_DIR.parent.parent / "envs" / "objects" / "industry_objects.py"

# Specify 'x', 'y' for flat side-lays, OR 'invert_x', 'invert_y' for upside-down flips
TARGET_OBJECTS = {
    #'thomas':"y",
    #'porter':"y",
    'mario':"x",
    #'mini_fire_engine':"y",
    #'mini_roller':"y",
    #'school_bus':"y",
    #'grimlock':"y",
    #'dragon':"y",
    #'bumblebee':"x",  
#    "mouse_naga": "invert_x",    # <--- NEW: Flips 180 degrees onto its back
}
# ==========================================

def calculate_rotated_bounds(obj_dir, xml_root, axis):
    """
    Parses all .obj meshes, applies the exact mathematical rotation, 
    and calculates the new bounding box and true maximum horizontal radius vertex.
    """
    min_z, max_z = float('inf'), float('-inf')
    max_r_sq = -1
    best_x, best_y = 0.0, 0.0
    mesh_files = []
    
    for asset in xml_root.findall(".//mesh"):
        file_attr = asset.get("file")
        if file_attr: 
            mesh_files.append(os.path.join(obj_dir, file_attr))

    if not mesh_files:
        return -0.02, 0.02, "0.05000 0 0"

    for mesh_path in mesh_files:
        if not os.path.exists(mesh_path): 
            continue
        with open(mesh_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                        # Simulate the 3D Rotation Math
                        if axis.lower() == 'y': # Pitch 90
                            nx, ny, nz = z, y, -x
                        elif axis.lower() == 'x': # Roll 90
                            nx, ny, nz = x, -z, y
                        elif axis.lower() == 'invert_x': # Roll 180 (Upside Down)
                            nx, ny, nz = x, -y, -z
                        elif axis.lower() == 'invert_y': # Pitch 180 (Upside Down)
                            nx, ny, nz = -x, y, -z
                        else: # Upright (or no rotation)
                            nx, ny, nz = x, y, z

                        min_z = min(min_z, nz)
                        max_z = max(max_z, nz)
                        
                        r_sq = nx**2 + ny**2
                        if r_sq > max_r_sq:
                            max_r_sq = r_sq
                            best_x = nx
                            best_y = ny

    if min_z == float('inf'):
        return -0.02, 0.02, "0.05000 0 0"

    radius_pos = f"{best_x:.5f} {best_y:.5f} 0"
    return min_z, max_z, radius_pos

def update_registry(class_name, snippet, registry_path):
    if not os.path.exists(registry_path):
        print(f"  [Registry Error] Could not find {registry_path}")
        return

    with open(registry_path, 'r') as f:
        content = f.read()

    pattern = rf"(?:@register_object\s+)?class {class_name}\(IndustryObject\):.*?(?=\n@register_object|\Z)"

    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, snippet.strip(), content, flags=re.DOTALL)
        print(f"  [Registry] Rewrote existing class: {class_name}")
    else:
        content += "\n\n" + snippet.strip() + "\n"
        print(f"  [Registry] Appended new class: {class_name}")

    with open(registry_path, 'w') as f:
        f.write(content)

def create_variants_and_register(obj_name, rot_axis, base_dir, registry_path):
    obj_dir = os.path.join(base_dir, obj_name)
    if not os.path.exists(obj_dir):
        print(f"  [Error] Directory not found: {obj_dir}")
        return

    # Determine if we are making a "flat" or "inverted" variant
    is_inverted = 'invert' in rot_axis.lower()
    variant_suffix = "inverted" if is_inverted else "flat"

    orig_path = os.path.join(obj_dir, f"{obj_name}.xml")
    upright_path = os.path.join(obj_dir, f"{obj_name}_upright.xml")
    variant_path = os.path.join(obj_dir, f"{obj_name}_{variant_suffix}.xml")

    # 1. Rename original to _upright.xml
    if os.path.exists(orig_path):
        os.rename(orig_path, upright_path)
        print(f"  [File] Renamed {obj_name}.xml -> {obj_name}_upright.xml")
    elif not os.path.exists(upright_path):
        print(f"  [Warning] Neither {obj_name}.xml nor {obj_name}_upright.xml found. Skipping.")
        return

    # 2. Parse the upright XML
    tree = ET.parse(upright_path)
    root = tree.getroot()

    # Fix the Upright XML Sites before proceeding
    u_min_z, u_max_z, u_radius_pos = calculate_rotated_bounds(obj_dir, root, 'z') 
    
    for site in root.findall(".//site"):
        name = site.get("name")
        if name == "horizontal_radius_site":
            site.set("pos", u_radius_pos)
        elif name == "bottom_site":
            site.set("pos", f"0 0 {u_min_z:.5f}")
        elif name == "top_site":
            site.set("pos", f"0 0 {u_max_z:.5f}")
            
    tree.write(upright_path, encoding="utf-8", xml_declaration=False)
    print(f"  [Fixed] Corrected true boundaries in {os.path.basename(upright_path)}")

    # 3. Calculate rotated bounds for the Variant XML (Flat or Inverted)
    min_z, max_z, radius_pos = calculate_rotated_bounds(obj_dir, root, rot_axis)

    # Extract blocks for the new XML
    default_block = root.find("default")
    default_xml = ET.tostring(default_block, encoding='unicode').strip() + "\n" if default_block is not None else ""

    asset_block = root.find("asset")
    asset_xml = ""
    if asset_block is not None:
        for child in asset_block:
            child_str = ET.tostring(child, encoding='unicode').strip()
            asset_xml += f"    {child_str}\n"

    size_block = root.find("size")
    size_xml = ET.tostring(size_block, encoding='unicode').strip() + "\n" if size_block is not None else ""

    object_body = root.find(".//body[@name='object']")
    inner_xml = ""
    
    if object_body is not None:
        if "euler" in object_body.attrib: del object_body.attrib["euler"]
        if "quat" in object_body.attrib: del object_body.attrib["quat"]

        # Apply the specific rotation quaternion
        if rot_axis.lower() == 'x':
            variant_quat = "0.7071068 0.7071068 0 0" # Roll 90 (Side flat)
        elif rot_axis.lower() == 'y':
            variant_quat = "0.7071068 0 0.7071068 0" # Pitch 90 (Front/Back flat)
        elif rot_axis.lower() == 'invert_x':
            variant_quat = "0 1 0 0" # Roll 180 (Upside Down)
        elif rot_axis.lower() == 'invert_y':
            variant_quat = "0 0 1 0" # Pitch 180 (Upside Down)
        else:
            variant_quat = "1 0 0 0" 

        # Strip old rotations from geoms/sites, but do NOT apply variant_quat here
        for elem in object_body.iter():
            if elem.tag in ['geom', 'site']:
                if "euler" in elem.attrib:
                    del elem.attrib["euler"]
                if "quat" in elem.attrib:
                    del elem.attrib["quat"]

        for child in object_body:
            child_str = ET.tostring(child, encoding='unicode').strip()
            inner_xml += f"        {child_str}\n"

    # Build the Variant XML
    variant_xml_content = f"""<mujoco model="{obj_name}_{variant_suffix}">
  {size_xml}
  {default_xml}
  <asset>
{asset_xml}  </asset>

  <worldbody>
    <body>
     <body name="object" pos="0 0 0" quat="{variant_quat}">
{inner_xml}      </body>
      
      <site rgba="0 0 1 1" size="0.01" pos="0 0 {min_z:.5f}" quat="1 0 0 0" name="bottom_site"/>  
      <site rgba="0 1 0 1" size="0.01" pos="0 0 {max_z:.5f}" quat="1 0 0 0" name="top_site"/>   
      <site rgba="1 0 0 1" size="0.01" pos="{radius_pos}" quat="1 0 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    with open(variant_path, 'w') as f:
        f.write(variant_xml_content)
    print(f"  [Generated] {obj_name}_{variant_suffix}.xml (Axis: {rot_axis.upper()})")
    print(f"  [Dimensions] Bottom Z: {min_z:.4f}, Top Z: {max_z:.4f}, Radius Pos: {radius_pos}")

    # 4. Generate Python Registry Snippets
    pascal_name = "".join(word.capitalize() for word in obj_name.split('_'))
    class_suffix = "Inverted" if is_inverted else "Flat"

    orig_z_offset = abs(u_max_z) if u_max_z != float('-inf') else 0.05

    upright_snippet = f"""
@register_object
class {pascal_name}Upright(IndustryObject):
    def __init__(
        self, 
        name="{obj_name}_upright", 
        obj_name="{obj_name}", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="{obj_name}_upright", joints=joints)
        self.rotation = {{"x": (0, 0), "y": (0, 0), "z": (0, 0)}}
        self.rotation_axis = 'z'
        self.z_offset = {orig_z_offset:.4f}
"""

    variant_snippet = f"""
@register_object
class {pascal_name}{class_suffix}(IndustryObject):
    def __init__(
        self, 
        name="{obj_name}_{variant_suffix}", 
        obj_name="{obj_name}", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="{obj_name}_{variant_suffix}", joints=joints)
        self.rotation = {{"x": (0, 0), "y": (0, 0), "z": (0, 0)}}
        self.rotation_axis = None
        self.z_offset = 0.005
"""

    # 5. Patch the Registry File
    update_registry(f"{pascal_name}Upright", upright_snippet, registry_path)
    update_registry(f"{pascal_name}{class_suffix}", variant_snippet, registry_path)


if __name__ == "__main__":
    print("Starting Object Processing & Registration Pipeline...")
    for obj, axis in TARGET_OBJECTS.items():
        print(f"\n--- Processing: {obj} on {axis.upper()}-Axis ---")
        create_variants_and_register(obj, axis, BASE_DIR, REGISTRY_FILE)
    print("\n✅ Pipeline Complete!")

import os
import re
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# OPTIONS: "upright", "flat", "inlay"
VIEW_MODE = "flat" 

TARGET_OBJECTS = [    
    'thomas',
    'porter',
    'mario',
    'mini_fire_engine',
    'mini_roller',
    'school_bus',
    # 'grimlock',
    # 'dragon',
    # 'bumblebee',
    ]

# Grid layout settings
SPACING = 0.15  # Distance between objects in meters
# ==========================================

def get_xml_path(obj_name, mode):
    """Maps the object name and mode to the correct file path."""
    if mode == "upright":
        return os.path.join(BASE_DIR, obj_name, f"{obj_name}_upright.xml")
    elif mode == "flat":
        return os.path.join(BASE_DIR, obj_name, f"{obj_name}_flat.xml")
    elif mode == "inlay":
        path = os.path.join(BASE_DIR, f"{obj_name}_upright_inlay", f"{obj_name}_upright_inlay.xml")
        if not os.path.exists(path):
            path = os.path.join(BASE_DIR, f"{obj_name}_flat_inlay", f"{obj_name}_flat_inlay.xml")
        return path
    elif mode == "inverted":
        return os.path.join(BASE_DIR, obj_name, f"{obj_name}_inverted.xml")
    return None

def sanitize_xml(content, index):
    """
    Dynamically appends the object index to all identifying attributes.
    This prevents ANY repeated name errors (like 'center_mark', 'model', etc.)
    """
    # These are the attributes MuJoCo uses to define and reference objects
    attributes_to_suffix = ['name', 'mesh', 'material', 'texture', 'class']
    
    for attr in attributes_to_suffix:
        # Regex: Find attr="<anything>" and replace with attr="<anything>_index"
        pattern = rf'{attr}="([^"]+)"'
        replacement = rf'{attr}="\1_{index}"'
        content = re.sub(pattern, replacement, content)
        
    return content

def launch_unified_viewer():
    print(f"--- Preparing Unified View: {VIEW_MODE.upper()} ---")
    
    
    combined_defaults = ""
    combined_assets = ""
    combined_worldbody = ""
    
    for i, obj_name in enumerate(TARGET_OBJECTS):
        xml_path = get_xml_path(obj_name, VIEW_MODE)
        if not xml_path or not os.path.exists(xml_path):
            print(f"  [Skip] File not found: {xml_path}")
            continue

        with open(xml_path, 'r') as f:
            content = f.read()

        # Safely auto-rename all custom names, meshes, and materials
        content = sanitize_xml(content, i)
        
        # Inject absolute folder paths so MuJoCo can find the .obj meshes
        folder_path = os.path.dirname(xml_path)
        content = content.replace('file="', f'file="{folder_path}/')

        # Parse the sanitized XML to safely extract components
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            print(f"  [Error] XML Parsing failed for {xml_path}: {e}")
            continue
            
        # 1. Extract Defaults
        def_block = root.find("default")
        if def_block is not None:
            for child in def_block:
                combined_defaults += ET.tostring(child, encoding='unicode') + "\n"

        # 2. Extract Assets (Meshes, Materials, Textures)
        asset_block = root.find("asset")
        if asset_block is not None:
            for child in asset_block:
                combined_assets += ET.tostring(child, encoding='unicode') + "\n"

        # 3. Extract the actual physical bodies from the Worldbody
        wb_block = root.find("worldbody")
        inner_body = ""
        if wb_block is not None:
            for child in wb_block:
                inner_body += ET.tostring(child, encoding='unicode') + "\n"

        # Wrap the extracted bodies in a spaced container
        combined_worldbody += f"""
    <body name="container_{i}" pos="{i * SPACING} 0 0">
        <site name="label_{i}" pos="0 0 0.15" type="sphere" size="0.005" rgba="1 1 1 1"/>
        {inner_body}
    </body>"""

    # Assemble the final, valid MuJoCo XML
    full_xml = f"""
<mujoco>
    <compiler angle="radian" meshdir="/"/>
    <default>
        {combined_defaults}
    </default>
    <asset>
        {combined_assets}
    </asset>
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
        {combined_worldbody}
    </worldbody>
</mujoco>"""

    try:
        model = mujoco.MjModel.from_xml_string(full_xml)
        data = mujoco.MjData(model)
        print(f"Success! Displaying {len(TARGET_OBJECTS)} objects in {VIEW_MODE} mode.")
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"MuJoCo Loading Error: {e}")

if __name__ == "__main__":
    launch_unified_viewer()

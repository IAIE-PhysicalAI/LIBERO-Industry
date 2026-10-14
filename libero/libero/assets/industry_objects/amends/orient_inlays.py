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

TARGET_OBJECTS = [
    "mario",
    "mini_fire_engine",
    "mini_roller",
    "porter",
    "thomas",
    "school_bus"
]

# TOGGLE: Set to True to drop objects, False to view only inlays
SPAWN_OBJECTS = True  

# Toggle this to view different pairings!
POSE = "flat"  # "upright" or "flat"

# Choose "inlay" or an exact groove suffix present under industry_objects:
# "v_groove", "flat_groove", or "upright_groove".
GROOVE_TYPE = "flat_groove" 

SPACING = 0.2
# ==========================================

def sanitize_xml(content, index):
    attributes_to_suffix = ['name', 'mesh', 'material', 'texture', 'class']
    for attr in attributes_to_suffix:
        pattern = rf'\b{attr}="([^"]+)"'
        replacement = rf'{attr}="\1_{index}"'
        content = re.sub(pattern, replacement, content)
    return content

def extract_components(xml_path, suffix_index):
    if not xml_path or not os.path.exists(xml_path):
        return "", "", ""
        
    with open(xml_path, 'r') as f:
        content = f.read()

    content = sanitize_xml(content, suffix_index)
    folder_path = os.path.dirname(xml_path)
    content = content.replace('file="', f'file="{folder_path}/')

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        return "", "", ""

    defaults, assets, worldbodies = "", "", ""
    
    def_block = root.find("default")
    if def_block is not None:
        for child in def_block: defaults += ET.tostring(child, encoding='unicode') + "\n"

    asset_block = root.find("asset")
    if asset_block is not None:
        for child in asset_block: assets += ET.tostring(child, encoding='unicode') + "\n"

    wb_block = root.find("worldbody")
    if wb_block is not None:
        for child in wb_block: worldbodies += ET.tostring(child, encoding='unicode') + "\n"

    return defaults, assets, worldbodies

def launch_unified_inlays():
    print(f"\n--- Preparing Unified View: {'Objects + ' if SPAWN_OBJECTS else ''}{POSE.upper()} {GROOVE_TYPE.upper()} ---")
    
    combined_defaults = ""
    combined_assets = ""
    combined_worldbody = ""

    for i, obj in enumerate(TARGET_OBJECTS):
        print(f"\nProcessing [{obj}]:")
        
        # 1. Determine Object XML Path
        obj_xml = os.path.join(BASE_DIR, obj, f"{obj}_{POSE}.xml")

        # 2. Determine the XML path from the asset's exact folder suffix.
        # V-groove assets are named <object>_v_groove even when POSE is "flat";
        # <object>_flat_v_groove does not exist for these target objects.
        if GROOVE_TYPE == "inlay":
            asset_suffix = f"{POSE}_inlay"
        elif GROOVE_TYPE in {"v_groove", "flat_groove", "upright_groove"}:
            asset_suffix = GROOVE_TYPE
        else:
            raise ValueError(
                "GROOVE_TYPE must be one of: inlay, v_groove, "
                "flat_groove, upright_groove"
            )

        folder_name = f"{obj}_{asset_suffix}"
            
        groove_xml = os.path.join(BASE_DIR, folder_name, f"{folder_name}.xml")

        # Load Inlay Components
        g_def, g_ass, g_wb = extract_components(groove_xml, f"grv_{i}")
        if g_wb:
            print(f"  [SUCCESS] Inlay spawned from {os.path.basename(groove_xml)}")
        else:
            print(f"  [ERROR] Inlay NOT found at {groove_xml}")

        # Load Object Components (if toggled)
        o_wb_wrapped = ""
        if SPAWN_OBJECTS:
            o_def, o_ass, o_wb = extract_components(obj_xml, f"obj_{i}")
            if o_wb:
                print(f"  [SUCCESS] Object spawned from {os.path.basename(obj_xml)}")
                combined_defaults += o_def
                combined_assets += o_ass
                
                x_offset = i * SPACING
                o_wb_wrapped = f"""
            <body name="drop_wrapper_{i}" pos="{x_offset} 0 0.15" euler="0 0 0">
                <inertial pos="0 0 0" mass="0.1" diaginertia="1e-3 1e-3 1e-3" />
                <freejoint name="free_{i}"/>
                {o_wb}
            </body>"""
            else:
                print(f"  [ERROR] Object NOT found at {obj_xml}")

        combined_defaults += g_def
        combined_assets += g_ass
        
        x_offset = i * SPACING
        combined_worldbody += f"""
    <body name="container_{i}" pos="{x_offset} 0 0">
        <site name="label_{i}" pos="0 0 0.15" type="sphere" size="0.005" rgba="1 0 0 1"/>
        {g_wb}
    </body>
    {o_wb_wrapped}"""

    full_xml = f"""
<mujoco>
    <compiler angle="degree" /> 
    <option viscosity="0.5" />
    <default>
        {combined_defaults}
    </default>
    <asset>
        {combined_assets}
    </asset>
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.4 0.4 0.4"/>
        {combined_worldbody}
    </worldbody>
</mujoco>"""

    try:
        model = mujoco.MjModel.from_xml_string(full_xml)
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"\nMuJoCo Loading Error:\n{e}")

if __name__ == "__main__":
    launch_unified_inlays()

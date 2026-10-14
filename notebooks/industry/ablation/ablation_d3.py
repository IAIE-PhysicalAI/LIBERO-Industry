import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
DRY_RUN = False  # Set to False to write the BDDL files

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Read from your newly generated Level 3 dataset
SOURCE_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "../../../libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d4"
))
# Write to Level 2 (Frozen Conveyors)
TARGET_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "../../../libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d3"
))

def update_language(lang_str):
    """Updates instruction language to reflect the stationary physics."""
    lang_str = lang_str.replace("moving tray", "tray")
    lang_str = lang_str.replace("the conveyor", "the conveyor")
    lang_str = lang_str.replace("stationary stationary", "stationary") 
    return lang_str

def process_stage_02_bddl(source_path, target_dir):
    with open(source_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    new_lang = ""
    original_filename = os.path.basename(source_path)
    target_ablation_region = "ablation_region"
    if "_D2S_" in original_filename:
        target_ablation_region = "ablation_region_d2s"
    elif "_D2D_" in original_filename:
        target_ablation_region = "ablation_region_d2d"
    has_target_ablation_region = any(
        line.strip() == f"({target_ablation_region}" for line in lines
    )

    for line in lines:
        stripped = line.strip()

        # --- Block 1: Update Language ---
        if stripped.startswith("(:language"):
            match = re.search(r'\(:language\s+(.*?)\)', line)
            if match:
                orig_lang = match.group(1)
                new_lang = update_language(orig_lang)
                line = f"  (:language {new_lang})\n"

        # --- Block 2: Freeze the Conveyor Belts ---
        # Updated regex to correctly capture negative velocities (e.g., -0.0286)
        elif stripped.startswith("(AtSpeed"):
            line = re.sub(r'(\(AtSpeed\s+\S+\s+)\S+', r'\g<1>0.0', line)
            
        # --- Block 3: Freeze the Ghost Tray Spawning Pulse ---
        elif stripped.startswith("(LoadSlot"):
            line = re.sub(
                r'(\(LoadSlot\s+\S+\s+\S+\s+\S+\s+\S+\s+)\S+',
                r'\g<1>0.0',
                line,
            )
            
        # --- Block 4: Assign Initial Object Placement ---
        # Identifies the object spawn line and locks it to the correct coordinate offset
        elif stripped.startswith("(On ablation_") and "conveyor_belt_1" in line:
            line = re.sub(r'conveyor_belt_1_[a-zA-Z0-9_]+', f'conveyor_belt_1_{target_ablation_region}', line)

        # --- Block 5: Inject BDDL Region Declaration ---
        # Finds the standard contact_region and injects the new region above it to satisfy the parser
        if stripped == "(contact_region" and not has_target_ablation_region:
            new_lines.append(f"      ({target_ablation_region}\n")
            new_lines.append(f"          (:target conveyor_belt_1)\n")
            new_lines.append(f"      )\n")

        new_lines.append(line)

    # --- Construct New Filename ---
    base_name = original_filename.replace("ABLD4", "ABLD3")
    target_filename = base_name.replace("moving_tray", "stationary_tray")

    target_path = os.path.join(target_dir, target_filename)

    if DRY_RUN:
        print(f"\n--- DRY RUN: {original_filename} -> {target_filename} ---")
        for orig, new in zip(lines, new_lines):
            if orig != new:
                print(f"[-] {orig.rstrip()}")
                print(f"[+] {new.rstrip()}")
    else:
        os.makedirs(target_dir, exist_ok=True)
        with open(target_path, 'w', encoding='utf-8', newline='') as f:
            f.writelines(new_lines)
        print(f"[SUCCESS] Wrote: {target_filename}")

if __name__ == "__main__":
    raise SystemExit("Use regenerate_ablation_flow.py with an empty --output-root.")

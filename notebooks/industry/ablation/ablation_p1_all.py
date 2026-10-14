import os
import re
import glob
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# ==========================================
# CONFIGURATION
# Toggle this variable between "p1" and "pt1"
# ==========================================
RUN_MODE = "p1"  

def generate_stage1_bddl():
    # 1. Define Paths based on RUN_MODE
    base_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision"
    
    if RUN_MODE == "p1":
        source_dir = os.path.join(base_dir, "libero_ablation_p0")
        target_dir_exact = os.path.join(base_dir, "libero_ablation_p1")
        target_dir_hold = os.path.join(base_dir, "libero_ablation_p1_hold")
        base_prefix = "ABL"       
        starting_number = 100     
    elif RUN_MODE == "pt1":
        source_dir = os.path.join(base_dir, "libero_ablation_pt0")
        target_dir_exact = os.path.join(base_dir, "libero_ablation_pt1")
        target_dir_hold = os.path.join(base_dir, "libero_ablation_pt1_hold")
        base_prefix = "ABLPT"     
        starting_number = 100     
    else:
        print("Invalid RUN_MODE. Please set to 'p1' or 'pt1'.")
        return

    # 2. Ensure target directories exist and clear old files
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    print(f"[{RUN_MODE.upper()} MODE] Copying BDDL files from {source_dir}...")
    for t_dir in [target_dir_exact, target_dir_hold]:
        os.makedirs(t_dir, exist_ok=True)
        for existing_file in glob.glob(os.path.join(t_dir, "*")):
            os.remove(existing_file)

    # 3. Copy files with sequential numbering
    source_bddl_files = sorted([
        f for f in os.listdir(source_dir)
        if f.endswith(".bddl") and os.path.isfile(os.path.join(source_dir, f))
    ])
    
    if len(source_bddl_files) == 0:
        print("No BDDL files found in the source directory.")
        return
    
    for idx, file_name in enumerate(source_bddl_files):
        current_number = starting_number + idx
        full_source_file = os.path.join(source_dir, file_name)
        
        clean_filename = re.sub(r'^(ABLPT|ABL)\d+H?_', '', file_name)
        
        # Exact
        exact_file_name = f"{base_prefix}{current_number:03d}_{clean_filename}"
        shutil.copy(full_source_file, os.path.join(target_dir_exact, exact_file_name))
        
        # Hold
        hold_file_name = f"{base_prefix}H{current_number:03d}_{clean_filename}"
        shutil.copy(full_source_file, os.path.join(target_dir_hold, hold_file_name))
    
    print(f"Copy and renaming complete for both Exact and Hold folders.\n")

    # ---------------------------------------------------------------------
    # PIPELINE A: Process Exact Directory (ReleasedOn)
    # ---------------------------------------------------------------------
    exact_bddl_files = sorted(glob.glob(os.path.join(target_dir_exact, "*.bddl")))
    print(f"Beginning EXACT conversion in {os.path.basename(target_dir_exact)}...\n")
    
    for i, filepath in enumerate(exact_bddl_files):
        with open(filepath, 'r') as f:
            content = f.read()
            
        # 1. Update Goal to ReleasedOn, strictly targeting the original plate_1
        content = re.sub(
            r'\(And\s+\(On\s+([a-zA-Z0-9_]+)\s+plate_1\)\)', 
            r'(And (ReleasedOn \1 plate_1))', 
            content
        )
            
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"[Exact {i+1}/{len(exact_bddl_files)}] CONVERTED: {os.path.basename(filepath)}")

    # ---------------------------------------------------------------------
    # PIPELINE B: Process Hold Directory (AllowHoldOn)
    # ---------------------------------------------------------------------
    print(f"\nBeginning HOLD conversion in {os.path.basename(target_dir_hold)}...\n")
    hold_bddl_files = sorted(glob.glob(os.path.join(target_dir_hold, "*.bddl")))

    for i, filepath in enumerate(hold_bddl_files):
        with open(filepath, 'r') as f:
            content = f.read()
            
        # 1. Update Goal to AllowHoldOn, strictly targeting the original plate_1
        content = re.sub(
            r'\(And\s+\(On\s+([a-zA-Z0-9_]+)\s+plate_1\)\)', 
            r'(And (AllowHoldOn \1 plate_1))', 
            content
        )
            
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"[Hold  {i+1}/{len(hold_bddl_files)}] CONVERTED: {os.path.basename(filepath)}")

    print("\nStage 1 generation completed successfully.")

if __name__ == "__main__":
    generate_stage1_bddl()

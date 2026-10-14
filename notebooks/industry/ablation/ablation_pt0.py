import os
import re
import glob
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

def amend_bddl_files():
    # Source and Target directories
    source_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision/libero_ablation_p0"
    target_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision/libero_ablation_pt0"
    
    # 1. Ensure target directory exists and copy files
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Copying BDDL files from {source_dir} to {target_dir}...")
    # Clear existing files in target dir to prevent cross-contamination on re-runs
    for existing_file in glob.glob(os.path.join(target_dir, "*")):
        os.remove(existing_file)

    for file_name in os.listdir(source_dir):
        if file_name.endswith(".bddl"):
            full_source_file = os.path.join(source_dir, file_name)
            if os.path.isfile(full_source_file):
                # Intercept the filename and change the prefix
                new_file_name = file_name.replace("ABL00", "ABLPT00")
                full_target_file = os.path.join(target_dir, new_file_name)
                
                # Copy using the specific new target path
                shutil.copy(full_source_file, full_target_file)
    print("Copy complete.\n")
        
    # 2. Grab and sort all copied BDDL files to ensure deterministic mapping
    bddl_files = sorted(glob.glob(os.path.join(target_dir, "*.bddl")))
    
    if len(bddl_files) == 0:
        print("No BDDL files found in the target directory after copy.")
        return
        
    # The distribution mapping (4 Bowls, 3 Cream Cheese, 3 Wine Bottles)
    distribution = [
        ("butter", "butter"),           # 10
        ("butter", "butter"),           # 10
        ("akita_black_bowl", "akita black bowl"), # 3
        ("cream_cheese", "cream cheese"),         # 5
        ("butter", "butter"),           # 8
        ("cream_cheese", "cream cheese"),         # 7
        ("akita_black_bowl", "akita black bowl"), # 3
        ("akita_black_bowl", "akita black bowl"), # 4
        ("akita_black_bowl", "akita black bowl"), # 3
        ("cream_cheese", "cream cheese"),         # 7
    ]
    
    print(f"Found {len(bddl_files)} BDDL files. Beginning conversion...\n")
    
    for i, filepath in enumerate(bddl_files):
        if i >= len(distribution):
            break
            
        obj_class, obj_lang = distribution[i]
        
        # Leave the first 4 files completely untouched
        if obj_class == "akita_black_bowl":
            print(f"[{i+1}/10] SKIPPED (Kept original): {os.path.basename(filepath)}")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        # ---------------------------------------------------------
        # 1. Update Natural Language Instruction
        content = re.sub(r'akita black bowl', obj_lang, content, flags=re.IGNORECASE)
        
        # 2. Update the (:objects) Declaration Block
        # WARNING: This assumes there are exactly TWO bowls in the source BDDL.
        orig_decl = "akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl"
        new_decl = f"{obj_class}_1 - {obj_class}\n    akita_black_bowl_2 - akita_black_bowl"
        content = content.replace(orig_decl, new_decl)
        
        # 3. Update Instances in (:init), (:goal), and (:obj_of_interest)
        content = content.replace("akita_black_bowl_1", f"{obj_class}_1")
        # ---------------------------------------------------------
        
        # 4. Rename File
        old_filename = os.path.basename(filepath)
        new_filename = old_filename.replace("akita_black_bowl", obj_class).replace("black_bowl", obj_class)
        new_filepath = os.path.join(target_dir, new_filename)
        
        # Write the new BDDL and delete the old one
        with open(new_filepath, 'w') as f:
            f.write(content)
            
        os.remove(filepath)
        print(f"[{i+1}/10] CONVERTED: {old_filename} \n           -> {new_filename}")

if __name__ == "__main__":
    amend_bddl_files()

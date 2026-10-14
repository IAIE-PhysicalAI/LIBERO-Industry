import os
import re
import glob
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

def amend_bddl_files():
    # Source and Target directories
    source_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision/libero_ablation_p0"
    target_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision/ablation_test_grasp"
    
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
                # Intercept the filename and change the prefix to match the new test 
                new_file_name = file_name.replace("ABL00", "ABLTG00")
                full_target_file = os.path.join(target_dir, new_file_name)
                
                # Copy using the specific new target path
                shutil.copy(full_source_file, full_target_file)
    print("Copy complete.\n")
        
    # 2. Grab and sort all copied BDDL files to ensure deterministic mapping
    bddl_files = sorted(glob.glob(os.path.join(target_dir, "*.bddl")))
    
    if len(bddl_files) == 0:
        print("No BDDL files found in the target directory after copy.")
        return
        
    # The distribution mapping: 10 new test objects
    # Corrected 'bbq)sauce' to 'bbq_sauce' to prevent BDDL parsing errors.
    distribution = [
        ("alphabet_soup", "alphabet soup"),
        ("bbq_sauce", "bbq sauce"),
        ("butter", "butter"),
        ("chocolate_pudding", "chocolate pudding"),
        ("tomato_sauce", "tomato sauce"),
        ("ketchup", "ketchup"),
        ("milk", "milk"),
        ("orange_juice", "orange juice"),
        ("salad_dressing", "salad dressing"),
        ("cream_cheese", "cream cheese"),
    ]
    
    print(f"Found {len(bddl_files)} BDDL files. Beginning conversion...\n")
    
    for i, filepath in enumerate(bddl_files):
        if i >= len(distribution):
            break
            
        obj_class, obj_lang = distribution[i]
            
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
        print(f"[{i+1}/{len(distribution)}] CONVERTED: {old_filename} \n           -> {new_filename}")

if __name__ == "__main__":
    amend_bddl_files()

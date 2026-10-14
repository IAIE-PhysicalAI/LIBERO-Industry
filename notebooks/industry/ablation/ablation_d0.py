import os
import re
from pathlib import Path

# Define paths
REPO_ROOT = Path(__file__).resolve().parents[3]
input_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_precision/libero_ablation_p0"
output_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d0"

os.makedirs(output_dir, exist_ok=True)

def adapt_to_industry_workbench(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Amend problem definition
    content = re.sub(
        r'\(problem\s+[A-Za-z0-9_]+\)', 
        '(problem LIBERO_Industry_Workbench_D0_Ablation)', 
        content
    )
    # Amend targets
    content = content.replace('(:target main_table)', '(:target industry_workbench)')

    # Amend fixtures
    content = content.replace('main_table - table', 'industry_workbench - industry_workbench')

    # Amend init states and region prefixes
    content = content.replace('main_table_', 'industry_workbench_')

    # Isolate original filename
    filename = os.path.basename(filepath)
    
    # Inject 'D' into the ABL00x prefix 
    # ^ targets the start of the string, (\d{3}) captures exactly 3 digits
    new_filename = re.sub(r'^ABL(\d{3})', r'ABLD\1', filename)
    
    output_filepath = os.path.join(output_dir, new_filename)
    
    with open(output_filepath, 'w') as f:
        f.write(content)
        
    print(f"  [+] Migrated env and saved as: {new_filename}")

    
# Execute over directory
print(f"Scanning directory: {input_dir}")

# Look for any .bddl files containing a 3-digit task code
target_pattern = re.compile(r'^.*?([0-9]{3})_.*\.bddl$')
matching_files = [f for f in os.listdir(input_dir) if target_pattern.match(f)]

if not matching_files:
    print("No matching BDDL files found in the input directory.")
else:
    print(f"Found {len(matching_files)} files. Beginning environment adaptation...\n")
    for file in matching_files:
        adapt_to_industry_workbench(os.path.join(input_dir, file))

    print(f"\nMigration complete. Files saved to {output_dir}")

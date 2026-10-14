import os
import re
from pathlib import Path

# Define paths
REPO_ROOT = Path(__file__).resolve().parents[3]
input_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d0"
output_dir = REPO_ROOT / "libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d1"

os.makedirs(output_dir, exist_ok=True)

# --- SPATIAL CONSTANTS ---
# Maps specific fixture regions to their local (Delta X, Delta Y) offsets 
# required to maintain exact global spawning coordinates after dependency flattening.
FIXTURE_OFFSETS = {
    'stove_region': (0.15, 0.0)
    # Add future fixtures here if their interaction sites are offset from their root.
}

def parse_bddl_blocks(content):
    """Extracts major BDDL blocks using regex."""
    blocks = {}
    for block_name in ['regions', 'fixtures', 'objects', 'obj_of_interest', 'init', 'goal']:
        pattern = rf'\(:{block_name}(.*?)\n  \)'
        match = re.search(pattern, content, re.DOTALL)
        blocks[block_name] = match.group(1) if match else ""
    return blocks

def extract_regions_list(region_str):
    """A parenthesis-counting parser to reliably isolate top-level region blocks."""
    regions = []
    depth = 0
    current_region = ""
    for char in region_str:
        if char == '(':
            depth += 1
            current_region += char
        elif char == ')':
            depth -= 1
            current_region += char
            if depth == 0 and current_region.strip():
                regions.append(current_region)
                current_region = ""
        else:
            if depth > 0:
                current_region += char
    return regions

def clean_bddl(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = parse_bddl_blocks(content)
    
    objs_of_interest = [obj.strip() for obj in blocks['obj_of_interest'].strip().split('\n') if obj.strip()]
    init_lines = [line.strip() for line in blocks['init'].strip().split('\n') if line.strip()]
    
    # --- AMENDMENT START: Flatten Placement Tree ---
    placement = {}
    for line in init_lines:
        # Capture relational placements: (On item base) or (In item base)
        m = re.search(r'\((On|In)\s+([^\s\)]+)\s+([^\s\)]+)\)', line)
        if m:
            rel, item, base = m.groups()
            placement[item] = (rel, base)

    # 1. Generate flattened init states strictly for objects of interest
    new_init = []
    for item in objs_of_interest:
        if item in placement:
            rel, base = placement[item]
            
            # Trace dependencies downward to the base region
            while True:
                # Rule A: Base is another standalone object (e.g., 'cookies_1')
                if base in placement and base not in objs_of_interest:
                    rel, base = placement[base]
                    continue
                
                # Rule B: Base is a region belonging to a fixture
                matched_parent = None
                for parent in placement.keys():
                    if base.startswith(parent + "_") and parent not in objs_of_interest:
                        matched_parent = parent
                        break
                
                if matched_parent:
                    rel, base = placement[matched_parent]
                    continue
                    
                break
            
            # [UPDATED FOR D0 ABLATION]: Sanitize relation targeting the industry workbench
            if base.startswith('industry_workbench_'):
                rel = 'On'
                
            new_init.append(f"    ({rel} {item} {base})")

    # 2. Filter objects block to strictly retain objects of interest
    new_objects = []
    for line in blocks['objects'].strip().split('\n'):
        if not line.strip(): continue
        parts = line.split(' - ')
        if len(parts) == 2:
            entities = parts[0].split()
            kept_entities = [e for e in entities if e in objs_of_interest]
            if kept_entities:
                new_objects.append(f"    {' '.join(kept_entities)} - {parts[1].strip()}")

    # 3. [UPDATED FOR D0 ABLATION]: Hardcode primary fixture (intermediaries are bypassed)
    new_fixtures = ["    industry_workbench - industry_workbench"]
    # --- AMENDMENT END ---

    # 6. Clean :regions 
    init_text_combined = "\n".join(new_init)
    goal_text_combined = blocks['goal'].strip()
    
    raw_regions = extract_regions_list(blocks['regions'])
    new_regions = []
    
    for region in raw_regions:
        name_match = re.match(r'\(\s*([a-zA-Z0-9_]+)', region)
        target_match = re.search(r'\(:target\s+([a-zA-Z0-9_]+)\)', region)
        
        if name_match and target_match:
            region_name = name_match.group(1)
            target_fixture = target_match.group(1)
            
            expected_init_reference = f"{target_fixture}_{region_name}"
            
            # Keeps region if invoked in our new flattened :init OR the goal
            if (expected_init_reference in init_text_combined) or (region_name in goal_text_combined):
                
                # Scrub the yaw rotation mathematically
                clean_region = re.sub(r'\s*\(:yaw_rotation\s*\(\s*\([^)]+\)\s*\)\s*\)', '', region)
                
                # --- Apply Spatial Offsets for Fixture Regions ---
                if region_name in FIXTURE_OFFSETS:
                    dx, dy = FIXTURE_OFFSETS[region_name]
                    
                    # Target the 4 coordinates inside the (:ranges ( (xmin ymin xmax ymax) )) block
                    # Apply linear translation (dx, dy) to maintain original global positions
                    clean_region = re.sub(
                        r'(\(:ranges\s*\(\s*\(\s*)(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)(\s*\)\s*\)\s*\))',
                        lambda m: f"{m.group(1)}{float(m.group(2)) + dx:.4f} {float(m.group(3)) + dy:.4f} {float(m.group(4)) + dx:.4f} {float(m.group(5)) + dy:.4f}{m.group(6)}",
                        clean_region
                    )
                
                new_regions.append("      " + clean_region.strip())

    # 7. Reconstruct the BDDL File & Clean Language
    header_match = re.search(r'(.*?)\(:regions', content, re.DOTALL)
    header = header_match.group(1) if header_match else content.split('(:regions')[0]

    lang_match = re.search(r'\(:language (.*?)\)', header)
    cleaned_lang = None
    if lang_match:
        original_lang = lang_match.group(1)
        cleaned_lang = re.sub(r'\s+(on|between|from|in|next to)\s+.*?(?=\s+and\s+place)', '', original_lang, flags=re.IGNORECASE)
        cleaned_lang = re.sub(r'\b([Pp]ick)\s+the\b', r'\1 up the', cleaned_lang)
        header = header.replace(original_lang, cleaned_lang)
    
    reconstructed = header + \
        "(:regions\n" + "\n".join(new_regions) + "\n    )\n\n" + \
        "  (:fixtures\n" + "\n".join(new_fixtures) + "\n  )\n\n" + \
        "  (:objects\n" + "\n".join(new_objects) + "\n  )\n\n" + \
        "  (:obj_of_interest\n    " + "\n    ".join(objs_of_interest) + "\n  )\n\n" + \
        "  (:init\n" + "\n".join(new_init) + "\n  )\n\n" + \
        "  (:goal\n  " + blocks['goal'].strip() + "\n  )\n\n)"

    # Write out with new prefix and dynamically updated language name.
    filename = os.path.basename(filepath)
    
    # Extract the last two digits (e.g., "00" from "ABLD000")
    id_match = re.search(r'^ABLD0([0-9]{2})', filename)
    task_idx = id_match.group(1) if id_match else "00"
    
    # Prepend a '1' to create the new ID (e.g., 00 -> 100)
    new_task_id = f"1{task_idx}"

    if cleaned_lang:
        safe_lang = cleaned_lang.strip().lower().replace(" ", "_")
        new_filename = f"ABLD{new_task_id}_{safe_lang}.bddl"  
    else:
        new_filename = re.sub(r'^ABLD0([0-9]{2})', rf'ABLD{new_task_id}', filename)
        
    output_filepath = os.path.join(output_dir, new_filename)
    
    with open(output_filepath, 'w') as f:
        f.write(reconstructed)
        
    print(f"  [+] Edited, flattened dependencies, and saved: {new_filename}")

# Execute over directory
print(f"Scanning directory: {input_dir}")

target_pattern = re.compile(r'^ABLD0[0-9]{2}_.*\.bddl$')
matching_files = [f for f in os.listdir(input_dir) if target_pattern.match(f)]

if not matching_files:
    print("No matching files found. Check your directory path and contents.")
else:
    print(f"Found {len(matching_files)} matching BDDL files. Beginning processing...\n")
    for file in matching_files:
        print(f"Reading file: {file}")
        clean_bddl(os.path.join(input_dir, file))

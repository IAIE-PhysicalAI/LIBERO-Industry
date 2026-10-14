import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure this points to the folder you want to analyze (e.g., d2s, d2d, or s2d)
BDDL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR,  "..", "..", "..", "libero/libero/bddl_files/libero_industry_d2s"))

def parse_bddl_for_table(bddl_path):
    with open(bddl_path, 'r') as f:
        content = f.read()

    task_name = os.path.basename(bddl_path).replace('.bddl', '')

    # 1. STRING PARTITIONING (Bug Fix)
    # Safely slice the BDDL into sections to avoid parenthesis truncation bugs
    objects_start = content.find('(:objects')
    init_start = content.find('(:init')
    goal_start = content.find('(:goal')

    objects_text = content[objects_start:init_start] if objects_start != -1 else ""
    init_text = content[init_start:goal_start] if init_start != -1 else ""
    goal_text = content[goal_start:] if goal_start != -1 else ""

    # 2. Extract All Objects (excluding types)
    all_objects = re.findall(r'([A-Za-z0-9_]+)\s+-', objects_text)

    # 3. Extract Spatial States ONLY
    # We strictly match spatial placement predicates, ignoring AtSpeed/LoadSlot logic
    spatial_pattern = re.compile(r'\(\s*(On|AlignedOn|Inside)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)', re.IGNORECASE)
    
    # regex returns tuples of (Predicate, Object, Region)
    raw_goal_matches = spatial_pattern.findall(goal_text)
    init_matches = spatial_pattern.findall(init_text)

    # Convert init matches into a quick-lookup set of (Object, Region) tuples
    init_pairs = set((match[1], match[2]) for match in init_matches)

    # 4. Logical Sorting Engine
    target_objects = []
    goal_regions = []
    target_confounders = []
    
    for _, obj, region in raw_goal_matches:
        if (obj, region) in init_pairs:
            # The goal condition is already met in the initial state!
            target_confounders.append(obj)
        else:
            # It's a true active target that needs to be moved
            target_objects.append(obj)
            goal_regions.append(region)

    # 5. Check for Initial Confounders
    raw_target_objs = [match[1] for match in raw_goal_matches]
    initial_confounders = [obj for obj in all_objects if obj not in raw_target_objs]

    # Clean up standard non-manipulable fixtures
    ignore_list = ["conveyor_belt_1", "conveyor_ghost_1", "grey_plate_1", "industry_workbench_1"]
    
    distractors = [
        obj for obj in (initial_confounders + target_confounders) 
        if obj not in ignore_list and "groove" not in obj and "inlay" not in obj
    ]

    # 6. Extract original instruction
    lang_match = re.search(r'\(:language\s+(.*?)\)', content)
    original_lang = lang_match.group(1) if lang_match else ""

    return {
        "Task ID": task_name,
        "Original Instruction": original_lang,
        "Target Objects": ", ".join(target_objects),
        "Goal Regions": ", ".join(goal_regions),
        "Distractor Objects": ", ".join(distractors),
        "Target Confounders": ", ".join(target_confounders)
    }

def main():
    data = []
    for f in os.listdir(BDDL_DIR):
        if f.endswith(".bddl"):
            data.append(parse_bddl_for_table(os.path.join(BDDL_DIR, f)))
    
    # Sort the dataframe by Task ID so Level 1, 2, and 3 are grouped cleanly
    df = pd.DataFrame(data)
    df = df.sort_values(by="Task ID")

    base_name = os.path.basename(BDDL_DIR)
    excel_filename = f"{base_name}.xlsx"
    
    df.to_excel(excel_filename, index=False)
    print(f"Exported perfectly parsed data to {excel_filename}")

if __name__ == "__main__":
    main()
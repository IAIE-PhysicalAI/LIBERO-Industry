import os
import re
import pandas as pd

# 1. Paths: Save in current directory, read from s2d
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BDDL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR,  "..", "..", "..", "libero/libero/bddl_files/libero_industry_s2d"))

def parse_s2d_bddl(bddl_path):
    with open(bddl_path, 'r') as f:
        content = f.read()

    task_name = os.path.basename(bddl_path).replace('.bddl', '')

    # 1. STRING PARTITIONING
    objects_start = content.find('(:objects')
    init_start = content.find('(:init')
    goal_start = content.find('(:goal')

    objects_text = content[objects_start:init_start] if objects_start != -1 else ""
    init_text = content[init_start:goal_start] if init_start != -1 else ""
    goal_text = content[goal_start:] if goal_start != -1 else ""

    # 2. Extract All Objects
    all_objects = re.findall(r'([A-Za-z0-9_]+)\s+-', objects_text)

    # 3. S2D Active Targets (From Goal Block)
    # S2D uses (Routed object conveyor_ghost_1)
    goal_pattern = re.compile(r'\(\s*Routed\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)', re.IGNORECASE)
    raw_goal_matches = goal_pattern.findall(goal_text)

    target_objects = [match[0] for match in raw_goal_matches]
    goal_regions = [match[1] for match in raw_goal_matches]

    # 4. S2D Target Confounders (From Init Block)
    # Any object already physically on the conveyor_ghost_1 before the task starts
    init_on_pattern = re.compile(r'\(\s*On\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)', re.IGNORECASE)
    init_on_matches = init_on_pattern.findall(init_text)

    target_confounders = []
    for obj, region in init_on_matches:
        # If the region is the conveyor ghost, it is a target confounder!
        if "conveyor_ghost" in region.lower() and obj in all_objects:
            if obj not in target_objects:  # Ensure it's not accidentally a target
                target_confounders.append(obj)

    # 5. S2D Initial Confounders
    initial_confounders = [
        obj for obj in all_objects 
        if obj not in target_objects and obj not in target_confounders
    ]

    # 6. Sieve Out Fixtures (Plates, Workbenches, Ghosts)
    distractors = []
    for obj in (initial_confounders + target_confounders):
        lower_obj = obj.lower()
        
        # S2D specific ignores (generalizes black_plate, grey_plate, etc.)
        if "plate" in lower_obj: continue
        if "conveyor" in lower_obj: continue
        if "workbench" in lower_obj: continue
        if "tray" in lower_obj: continue
        
        distractors.append(obj)

    # Refilter target confounders just to be safe and keep the excel clean
    target_confounders = [obj for obj in target_confounders if obj in distractors]

    # 7. Extract Language Instruction
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
    if not os.path.exists(BDDL_DIR):
        print(f"[!] Could not find S2D directory at: {BDDL_DIR}")
        return

    data = []
    for f in os.listdir(BDDL_DIR):
        if f.endswith(".bddl"):
            data.append(parse_s2d_bddl(os.path.join(BDDL_DIR, f)))
    
    # Sort dataframe cleanly by Task ID
    df = pd.DataFrame(data)
    df = df.sort_values(by="Task ID")

    # Save exactly in the same directory as this script
    base_name = os.path.basename(BDDL_DIR)
    excel_filename = os.path.join(SCRIPT_DIR, f"{base_name}.xlsx")
    
    df.to_excel(excel_filename, index=False)
    print(f"[SUCCESS] Exported S2D parsed data to: {excel_filename}")

if __name__ == "__main__":
    main()
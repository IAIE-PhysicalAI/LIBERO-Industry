import os
import re

# ==========================================
# CONFIGURATION - LEVEL 4
# ==========================================
DRY_RUN = False 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, 
    "../../../libero/libero/bddl_files/ablation_industry_flow/libero_original_lv1"
))
TARGET_DIR_L4 = os.path.normpath(os.path.join(
    SCRIPT_DIR, 
    "../../../libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d4"
))

BALANCED_TASKS = [
    # D2D (Dynamic-to-Dynamic) 4 tasks
    {"id": 0,  "type": "D2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 1,  "type": "D2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 2,  "type": "D2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 3,  "type": "D2D", "target_obj": "ablation_akita_black_bowl"},
    
    # D2S (Dynamic-to-Static) 4 tasks
    {"id": 4,  "type": "D2S", "target_obj": "ablation_akita_black_bowl"},
    {"id": 5,  "type": "D2S", "target_obj": "ablation_akita_black_bowl"},
    {"id": 6,  "type": "D2S", "target_obj": "ablation_akita_black_bowl"},
    {"id": 7,  "type": "D2S", "target_obj": "ablation_akita_black_bowl"},

    # S2D (Static-to-Dynamic) 4 tasks
    {"id": 8,  "type": "S2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 9,  "type": "S2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 10, "type": "S2D", "target_obj": "ablation_akita_black_bowl"},
    {"id": 11, "type": "S2D", "target_obj": "ablation_akita_black_bowl"},
    # {"id": 0,  "type": "D2D", "target_obj": "ablation_akita_black_bowl"},
    # {"id": 1,  "type": "D2D", "target_obj": "ablation_cream_cheese"},
    # {"id": 2,  "type": "D2D", "target_obj": "ablation_bbq_sauce"},
    # {"id": 3,  "type": "D2D", "target_obj": "ablation_tomato_sauce"},
    
    # # D2S (Dynamic-to-Static) 4 tasks
    # {"id": 4,  "type": "D2S", "target_obj": "ablation_akita_black_bowl"},
    # {"id": 5,  "type": "D2S", "target_obj": "ablation_cream_cheese"},
    # {"id": 6,  "type": "D2S", "target_obj": "ablation_bbq_sauce"},
    # {"id": 7,  "type": "D2S", "target_obj": "ablation_tomato_sauce"},

    # # S2D (Static-to-Dynamic) 4 tasks
    # {"id": 8,  "type": "S2D", "target_obj": "ablation_akita_black_bowl"},
    # {"id": 9,  "type": "S2D", "target_obj": "ablation_cream_cheese"},
    # {"id": 10, "type": "S2D", "target_obj": "ablation_bbq_sauce"},
    # {"id": 11, "type": "S2D", "target_obj": "ablation_tomato_sauce"},
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_language(new_name, task_type):
    templates = {
        "D2D": "put the {} from the conveyor into the moving tray",
        "D2S": "pick the {} from the conveyor and place it on the plate",
        "S2D": "put the {} to the moving tray",
    }
    return templates[task_type].format(new_name)

def extract_old_objects(lines):
    in_obj = False
    objs = []
    for line in lines:
        if line.strip() == "(:objects":
            in_obj = True
        elif in_obj and line.strip() == ")":
            break
        elif in_obj and "-" in line:
            left_side = line.split("-")[0]
            if not any(f in line for f in ["industry_workbench", "conveyor_belt", "conveyor_ghost", "grey_plate"]):
                objs.extend([w.strip() for w in left_side.split()])
    return objs

def get_valid_placements(lines, old_objs, is_s2d):
    valid = []
    for line in lines:
        if line.strip().startswith("(On "):
            for obj in old_objs:
                if f" {obj} " in line:
                    if is_s2d:
                        valid.append(obj)
                    else:
                        if "region" in line:
                            valid.append(obj)
    return list(dict.fromkeys(valid)) if valid else old_objs

def parse_goal_predicates(lines, old_objs):
    goal_map = {}
    in_goal = False
    goal_content = ""
    for line in lines:
        if line.strip() == "(:goal":
            in_goal = True
        elif in_goal and line.strip() == ")":
            break
        elif in_goal:
            goal_content += line

    preds = re.findall(r'\(([A-Za-z]+)\s+([^\s]+)\s+([^\s\)]+)\)', goal_content)
    for p, o, t in preds:
        if o in old_objs:
            goal_map[o] = (p, t)

    default_pred, default_tgt = "DynamicOn", "conveyor_ghost_1"
    if preds:
        default_pred, default_tgt = preds[0][0], preds[0][2]
    return goal_map, default_pred, default_tgt

# ==========================================
# MAIN PROCESSOR - D4
# ==========================================
def process_file(source_filepath, task_config):
    with open(source_filepath, 'r') as f:
        lines = f.readlines()
        
    original_filename = os.path.basename(source_filepath)
    is_s2d = task_config["type"] == "S2D"
    
    match = re.search(r'LV1_(\d+)', original_filename)
    orig_lv1_num = match.group(1) if match else "001"

    old_objs = extract_old_objects(lines)
    valid_objs = get_valid_placements(lines, old_objs, is_s2d)
    goal_map, default_pred, default_tgt = parse_goal_predicates(lines, old_objs)

    # 1. Setup Object Mapping (D4 Logic)
    object_mapping = {} 
    target_class = task_config["target_obj"]
    clean_name = target_class.replace("ablation_", "").replace("_", " ")
    
    kept_old_obj = valid_objs[0] 
    found_better_region = False
    
    # Canonical conveyor numbering is nearest-first: slot 1 reaches the
    # workstation first, followed by slots 2 and 3.
    preferred_regions = ["region_1", "region_2", "region_3"]
        
    for pref_region in preferred_regions:
        for line in lines:
            if line.strip().startswith("(On ") and (pref_region in line):
                for obj in valid_objs:
                    if f" {obj} " in line:
                        kept_old_obj = obj
                        found_better_region = True
                        break
            if found_better_region:
                break
        if found_better_region:
            break
            
    object_mapping[kept_old_obj] = (target_class, f"{target_class}_1")
    # 2. Extract and Update Language
    original_lang = ""
    for line in lines:
        if line.strip().startswith("(:language"):
            match = re.search(r'\(:language\s+(.*?)\)', line)
            if match:
                original_lang = match.group(1)
            break
            
    new_lang = clean_language(clean_name, task_config["type"])
    
    # 3. Output File Name Construction 
    prefix = "ABLD4"
    task_id = task_config["id"]
    t_type = task_config["type"]
    file_lang = new_lang.replace(' ', '_')
    
    new_filename = f"{prefix}{task_id:02d}_{t_type}_LV1_{orig_lv1_num}_{file_lang}.bddl"
    target_filepath = os.path.join(TARGET_DIR_L4, new_filename)

    # 4. Safely Rewrite Blocks
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if "define (problem" in stripped:
            new_lines.append(line.replace("LIBERO_Industry_Workbench_Manipulation", "LIBERO_Industry_Workbench_Ablation"))
            i += 1
            continue

        if stripped.startswith("(:language"):
            new_lines.append(f"  (:language {new_lang})\n")
            i += 1
            continue

        if stripped == "(:objects":
            new_lines.append("  (:objects\n")
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                if "-" in lines[i] and any(f in lines[i] for f in ["industry_workbench", "conveyor", "grey_plate"]):
                    new_lines.append(lines[i]) 
                i += 1
            for old_obj, (new_class, new_inst) in object_mapping.items():
                new_lines.append(f"    {new_inst} - {new_class}\n")
            new_lines.append("  )\n\n")
            i += 1
            continue

        if stripped == "(:obj_of_interest":
            new_lines.append("  (:obj_of_interest\n")
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                i += 1
            for old_obj, (new_class, new_inst) in object_mapping.items():
                new_lines.append(f"    {new_inst}\n")
            new_lines.append("  )\n\n")
            i += 1
            continue

        if stripped == "(:init":
            new_lines.append("  (:init\n")
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                init_line = lines[i]
                discarded = set(old_objs) - set(object_mapping.keys())
                if any(f" {d} " in init_line for d in discarded):
                    i += 1
                    continue
                mod_line = init_line
                for old_obj, (new_class, new_inst) in object_mapping.items():
                    if f" {old_obj} " in mod_line:
                        mod_line = mod_line.replace(f" {old_obj} ", f" {new_inst} ")
                        if not is_s2d:
                            mod_line = re.sub(
                                r'(offset|tail)_region_\d+',
                                lambda match: f"{match.group(1)}_bowl_3",
                                mod_line,
                            )
                new_lines.append(mod_line)
                i += 1
            new_lines.append("  )\n\n")
            i += 1
            continue

        if stripped == "(:goal":
            new_lines.append("  (:goal\n    (And")
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                i += 1
            for old_obj, (new_class, new_inst) in object_mapping.items():
                pred, target = default_pred, default_tgt
                if old_obj in goal_map:
                    pred, target = goal_map[old_obj]
                new_lines.append(f" ({pred} {new_inst} {target})")
            new_lines.append(")\n  )\n\n")
            i += 1
            continue

        new_lines.append(line)
        i += 1
             
    final_output = []
    for line in new_lines:
        if line.strip() != "":
            final_output.append(line)
        elif len(final_output) > 0 and final_output[-1].strip() != "":
            final_output.append("\n") 

    if not DRY_RUN:
        with open(target_filepath, 'w', encoding='utf-8', newline='') as f:
            f.writelines(final_output)
        print(f"[SUCCESS] Wrote: {new_filename}")

if __name__ == "__main__":
    raise SystemExit("Use regenerate_ablation_flow.py with an empty --output-root.")

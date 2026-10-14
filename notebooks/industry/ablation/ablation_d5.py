import os
import re
import random

# ==========================================
# CONFIGURATION - LEVEL 5 ONLY
# ==========================================
DRY_RUN = False

USE_BOWL_REGIONS = True
TARGET_OBJ_CLASS = "ablation_akita_black_bowl"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, 
    "../../../libero/libero/bddl_files/ablation_industry_flow/libero_original_lv1"
))

TARGET_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, 
    "../../../libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d5"
))

BALANCED_TASKS = [
    {"id": 0,  "type": "D2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 1,  "type": "D2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 2,  "type": "D2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 3,  "type": "D2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 4,  "type": "D2S", "target_obj": TARGET_OBJ_CLASS},
    {"id": 5,  "type": "D2S", "target_obj": TARGET_OBJ_CLASS},
    {"id": 6,  "type": "D2S", "target_obj": TARGET_OBJ_CLASS},
    {"id": 7,  "type": "D2S", "target_obj": TARGET_OBJ_CLASS},
    {"id": 8,  "type": "S2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 9,  "type": "S2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 10, "type": "S2D", "target_obj": TARGET_OBJ_CLASS},
    {"id": 11, "type": "S2D", "target_obj": TARGET_OBJ_CLASS},
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_language(new_name, task_type):
    plural_name = f"{new_name}s"
    templates = {
        "D2D": "put the {} from the conveyor into the moving tray",
        "D2S": "pick the {} from the conveyor and place them on the plate",
        "S2D": "put the {} to the moving tray",
    }
    return templates[task_type].format(plural_name)

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
# MAIN PROCESSOR
# ==========================================
def process_file(source_filepath, task_config):
    with open(source_filepath, 'r') as f:
        lines = f.readlines()
        
    original_filename = os.path.basename(source_filepath)
    is_s2d = task_config["type"] == "S2D"
    is_d2s = task_config["type"] == "D2S"
    
    match = re.search(r'LV1_(\d+)', original_filename)
    orig_lv1_num = match.group(1) if match else "001"

    old_objs = extract_old_objects(lines)
    valid_objs = get_valid_placements(lines, old_objs, is_s2d)

    # ---------------------------------------------------------
    # S2D INJECTION LOGIC
    # Synthetically generate a second object if only 1 exists
    # ---------------------------------------------------------
    if is_s2d and len(valid_objs) < 2:
        for idx, line in enumerate(lines):
            if f" {valid_objs[0]} " in line and line.strip().startswith("(On "):
                new_line = line.replace(valid_objs[0], "fake_s2d_obj")
                match = re.search(r'region_(\d+)', line)
                if match:
                    old_digit = int(match.group(1))
                    # Shift region by 3 to prevent object clipping on the plate
                    new_digit = (old_digit + 3) % 6
                    if new_digit == 0: new_digit = 6
                    new_line = re.sub(r'region_\d+', f'region_{new_digit}', new_line)
                
                lines.insert(idx + 1, new_line)
                valid_objs.append("fake_s2d_obj")
                old_objs.append("fake_s2d_obj")
                break

    goal_map, default_pred, default_tgt = parse_goal_predicates(lines, old_objs)

    # 1. Setup Exact 2-Object Mapping
    object_mapping = {} 
    target_class = task_config["target_obj"]
    clean_name = target_class.replace("ablation_", "").replace("_", " ")
    
    kept_old_objs = []
    # Canonical conveyor numbering is nearest-first: slot 1 reaches the
    # workstation first, followed by slots 2 and 3.
    preferred_regions = ["region_1", "region_2", "region_3"]
        
    for pref_region in preferred_regions:
        for line in lines:
            if line.strip().startswith("(On ") and (pref_region in line):
                for obj in valid_objs:
                    if f" {obj} " in line and obj not in kept_old_objs:
                        kept_old_objs.append(obj)
                        if len(kept_old_objs) == 2: break
            if len(kept_old_objs) == 2: break
        if len(kept_old_objs) == 2: break
            
    for obj in valid_objs:
        if len(kept_old_objs) == 2: break
        if obj not in kept_old_objs:
            kept_old_objs.append(obj)
            
    for i, old_obj in enumerate(kept_old_objs):
        object_mapping[old_obj] = (target_class, f"{target_class}_{i+1}")

    # 2. Extract and Update Language
    original_lang = ""
    for line in lines:
        if line.strip().startswith("(:language"):
            match = re.search(r'\(:language\s+(.*?)\)', line)
            if match:
                original_lang = match.group(1)
            break
            
    new_lang = clean_language(clean_name, task_config["type"])
    
    # 3. Output File Name Construction (Fused Prefix format: ABLD501)
    task_id = task_config["id"]
    t_type = task_config["type"]
    file_lang = new_lang.replace(' ', '_')
    
    new_filename = f"ABLD5{task_id:02d}_{t_type}_LV1_{orig_lv1_num}_{file_lang}.bddl"
    target_filepath = os.path.join(TARGET_DIR, new_filename)
    # ---> ADD THIS BLOCK HERE <---
    # D5 S2D Safe Patterns (Pick 2 non-adjacent from odds or evens)
    d5_s2d_safe_patterns = [(1, 3), (3, 5), (1, 5), (2, 4), (4, 6), (2, 6)]
    chosen_s2d_pattern = random.Random(task_id).choice(d5_s2d_safe_patterns)

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

        # ---------------------------------------------------------
        # REGIONS BLOCK INTERCEPT
        # ---------------------------------------------------------
        if stripped == "(:regions":
            new_lines.append("    (:regions\n")
            i += 1
            while i < len(lines):
                line_str = lines[i]
                stripped_line = line_str.strip()
                
                # 1. Filter out the old offset and tail regions if using bowls
                if USE_BOWL_REGIONS and re.search(r'\((offset_region|tail_region)', stripped_line):
                    # The original BDDL region declarations are exactly 3 lines long. 
                    # We skip the declaration, the (:target...), and the closing bracket.
                    i += 3
                    continue
                    
                # 2. Detect the end of the regions block
                if stripped_line == ")":
                    # Look ahead to verify this is the final closing bracket before fixtures
                    lookahead = i + 1
                    while lookahead < len(lines) and lines[lookahead].strip() == "":
                        lookahead += 1
                        
                    if lookahead < len(lines) and lines[lookahead].strip() == "(:fixtures":
                        if USE_BOWL_REGIONS:
                            bowl_regions = [
                                "offset_bowl_1", "offset_bowl_2", "offset_bowl_3",
                                "tail_bowl_1", "tail_bowl_2", "tail_bowl_3",
                            ]
                            declared = {candidate.strip() for candidate in lines}
                            for br in bowl_regions:
                                if f"({br}" not in declared:
                                    new_lines.append(f"      ({br}\n          (:target conveyor_belt_1)\n      )\n")
                        
                        new_lines.append("    )\n\n")
                        i += 1
                        break # Exit inner loop, proceed to (:fixtures

                new_lines.append(line_str)
                i += 1
            continue
        
        if stripped == "(:objects":
            new_lines.append("  (:objects\n")
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                if "-" in lines[i] and any(f in lines[i] for f in ["industry_workbench", "conveyor", "grey_plate"]):
                    new_lines.append(lines[i]) 
                i += 1
                
            # Group instances by their class (e.g., bowl_1 bowl_2 - bowl_class)
            class_groups = {}
            for old_obj, (new_class, new_inst) in object_mapping.items():
                if new_class not in class_groups:
                    class_groups[new_class] = []
                class_groups[new_class].append(new_inst)
                
            for new_class, instances in class_groups.items():
                instances_str = " ".join(instances)
                new_lines.append(f"    {instances_str} - {new_class}\n")
                
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
                        
                        # ---------------------------------------------------------
                        # D5 STRICT SPACING & XML RENAMING
                        # ---------------------------------------------------------
                        if is_s2d:
                            # S2D: Apply the safe, non-adjacent plate pattern
                            match = re.search(r'(place_region_|region_)(\d+)', mod_line)
                            if match:
                                prefix = match.group(1)
                                if new_inst.endswith("_1"): mod_line = mod_line.replace(match.group(0), f"{prefix}{chosen_s2d_pattern[0]}")
                                elif new_inst.endswith("_2"): mod_line = mod_line.replace(match.group(0), f"{prefix}{chosen_s2d_pattern[1]}")
                        else:
                            # Conveyor: Rename to "bowl" format and Map to 2 and 3
                            is_tail = "tail_" in mod_line
                            base_type = "tail" if is_tail else "offset"
                            new_prefix = f"{base_type}_bowl_"
                            
                            if new_inst.endswith("_1"): 
                                mod_line = re.sub(r'(offset|tail)_region_\d+', f'{new_prefix}2', mod_line)
                            elif new_inst.endswith("_2"): 
                                mod_line = re.sub(r'(offset|tail)_region_\d+', f'{new_prefix}3', mod_line)

                new_lines.append(mod_line)
                i += 1
            new_lines.append("  )\n\n")
            i += 1
            continue

        if stripped == "(:goal":
            new_lines.append("  (:goal\n    (And")
            for old_obj, (new_class, new_inst) in object_mapping.items():
                pred, target = default_pred, default_tgt
                if old_obj in goal_map:
                    pred, target = goal_map[old_obj]
                new_lines.append(f" ({pred} {new_inst} {target})")
                
            # Safely close the bracket structure and hard-break
            new_lines.append(")\n  )\n)\n")
            break

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

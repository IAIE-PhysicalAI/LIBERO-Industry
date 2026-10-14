import argparse
import csv
import os
import glob
import importlib.util
import re

# ==========================================
# CONFIGURATION
# ==========================================
DRY_RUN = True  # Set to False to write changes to the OUTPUT_PY_PATH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# Inputs: manually authored language tables.
CSV_FILES = [
    "d2s.csv",
    "s2d.csv",
    "d2d.csv",
]

# Validation: The BDDL directory to cross-reference
BDDL_DIR = os.path.join(LIBERO_ROOT, "libero", "libero", "bddl_files")

# Target directories for dynamic ablation instruction parsing
ABLATION_DIRS = [
    os.path.join(BDDL_DIR, "ablation_industry_flow"),
]

# Output: The target Python dictionary file
OUTPUT_PY_PATH = os.path.join(LIBERO_ROOT, "libero", "libero", "benchmark", "libero_suite_language_map.py")


def write_language_map(language_map):
    os.makedirs(os.path.dirname(OUTPUT_PY_PATH), exist_ok=True)
    with open(OUTPUT_PY_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("# ==================================================================\n")
        f.write("# Auto-generated flat language mapping for LIBERO-Industry\n")
        f.write("# Do not edit manually. Generate via notebooks/industry/language/language_map_helper.py\n")
        f.write("# ==================================================================\n\n")
        f.write("libero_language_map = {\n")
        for task_id in sorted(language_map):
            instruction = language_map[task_id].replace('"', '\\"')
            f.write(f'    "{task_id}": "{instruction}",\n')
        f.write("}\n")


def sync_ablation_flow(flow_root, dry_run=True):
    """Replace only D2-D6 language entries, leaving every other suite untouched."""
    generated = {}
    for tier in range(2, 7):
        suite_dir = os.path.join(flow_root, f"libero_ablation_d{tier}")
        paths = sorted(glob.glob(os.path.join(suite_dir, "*.bddl")))
        if len(paths) != 12:
            raise ValueError(f"Expected 12 BDDLs in {suite_dir}, found {len(paths)}")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            match = re.search(r"\(:language\s+(.*?)\)", text)
            if match is None:
                raise ValueError(f"Missing language declaration: {path}")
            generated[os.path.splitext(os.path.basename(path))[0]] = match.group(1)

    existing = load_existing_map(OUTPUT_PY_PATH)
    retained = {
        task: language
        for task, language in existing.items()
        if re.match(r"^ABLD[2-6]\d{2}_", task) is None
    }
    old_count = len(existing) - len(retained)
    retained.update(generated)
    print(
        f"Ablation D2-D6 language sync: remove={old_count}, add={len(generated)}, "
        f"total={len(retained)}, dry_run={dry_run}"
    )
    if not dry_run:
        write_language_map(retained)
    return retained

def get_valid_bddl_tasks():
    """Scans the core BDDL directory to find all valid task names."""
    valid_tasks = set()
    search_pattern = os.path.join(BDDL_DIR, "**", "*.bddl")
    
    for filepath in glob.glob(search_pattern, recursive=True):
        task_name = os.path.basename(filepath).replace(".bddl", "")
        valid_tasks.add(task_name)
        
    return valid_tasks

def extract_instruction_from_name(task_id):
    """
    Auto-extracts the natural language instruction from the BDDL filename.
    Bypasses uppercase prefixes and captures the lowercase text.
    """
    words = task_id.split('_')
    instruction_start_idx = 0
    
    for i, word in enumerate(words):
        if word.islower() and not word.isdigit():
            instruction_start_idx = i
            break

    if instruction_start_idx == 0:
        return task_id.replace("_", " ")

    return " ".join(words[instruction_start_idx:])

def load_existing_map(filepath):
    """Dynamically loads the existing language map to perform a state comparison."""
    if not os.path.exists(filepath):
        return {}
    
    try:
        spec = importlib.util.spec_from_file_location("existing_map", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "libero_language_map", {})
    except Exception as e:
        print(f"[!] Warning: Could not parse existing map for comparison. Assuming empty. Error: {e}")
        return {}

def generate_language_map():
    print(f"\n{'='*60}")
    print(f" BUILDING FLAT LIBERO LANGUAGE MAP | DRY_RUN: {DRY_RUN}")
    print(f"{'='*60}\n")
    
    valid_tasks = get_valid_bddl_tasks()
    print(f"[*] Found {len(valid_tasks)} valid .bddl files in physics backend.")
    
    language_map = {}
    dropped_tasks = []

    # =========================================================================
    # 1. AUTO-PROCESS ABLATION TASKS DIRECTLY FROM TARGET DIRECTORIES
    # =========================================================================
    ablation_count = 0
    for ablation_dir in ABLATION_DIRS:
        if not os.path.exists(ablation_dir):
            print(f"[!] Warning: Ablation directory not found: {ablation_dir}")
            continue
            
        search_pattern = os.path.join(ablation_dir, "libero_ablation_*", "*.bddl")
        for filepath in glob.glob(search_pattern):
            task_id = os.path.basename(filepath).replace(".bddl", "")
            instruction = extract_instruction_from_name(task_id)
            language_map[task_id] = instruction
            ablation_count += 1
            
    print(f"[*] Auto-generated instructions for {ablation_count} Ablation BDDL files.")

    # =========================================================================
    # 2. EXTRACT DATA FROM CSV FOR CORE TASKS
    # =========================================================================
    for csv_file in CSV_FILES:
        file_path = os.path.join(SCRIPT_DIR, csv_file)
        if not os.path.exists(file_path):
            print(f"[!] Warning: Could not find {csv_file}. Skipping...")
            continue

        with open(file_path, newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or "Task ID" not in rows.fieldnames:
                print(f"[!] Warning: {csv_file} has no Task ID column. Skipping...")
                continue

            for row in rows:
                task_id = str(row.get("Task ID", "")).strip()

                if not task_id or task_id.lower() == "nan":
                    continue

                if task_id not in valid_tasks:
                    dropped_tasks.append(task_id)
                    continue

                if task_id in language_map:
                    continue

                target_instruction = ""

                if "_LV1_" in task_id:
                    target_instruction = str(row.get("LV1_Instruction", "")).strip()
                elif "_LV2_" in task_id:
                    target_instruction = str(row.get("LV2_Instruction", "")).strip()
                elif "_LV3_" in task_id:
                    target_instruction = str(row.get("LV3_Instruction", "")).strip()

                if target_instruction and target_instruction.lower() != "nan":
                    clean_instruction = target_instruction.replace('"', '\\"')
                    language_map[task_id] = clean_instruction

    if dropped_tasks:
        print(f"\n[!] Dropped {len(dropped_tasks)} CSV rows because matching .bddl files do not exist.")

    # =========================================================================
    # 3. DRY RUN COMPARISON LOGIC (UPDATED)
    # =========================================================================
    existing_map = load_existing_map(OUTPUT_PY_PATH)
    
    new_entries = []
    updated_entries = []
    removed_entries = []
    unchanged_count = 0
    
    # Check for new and updated entries
    for task_id, instruction in language_map.items():
        if task_id not in existing_map:
            new_entries.append((task_id, instruction))
        elif existing_map[task_id] != instruction:
            updated_entries.append((task_id, existing_map[task_id], instruction))
        else:
            unchanged_count += 1

    # Check for removed entries (in old map, but not in current generation)
    for old_task, old_instruction in existing_map.items():
        if old_task not in language_map:
            removed_entries.append((old_task, old_instruction))

    print(f"\n{'='*60}")
    print(f" DIFF REPORT: {OUTPUT_PY_PATH}")
    print(f"{'='*60}")
    print(f" Total Output Target: {len(language_map)} tasks")
    print(f" Unchanged:           {unchanged_count}")
    print(f" New Additions:       {len(new_entries)}")
    print(f" Modified Overwrites: {len(updated_entries)}")
    print(f" Removed Deletions:   {len(removed_entries)}\n")

    if new_entries:
        print("[+] NEW ENTRIES TO BE ADDED:")
        for task, inst in new_entries[:5]:
            print(f"    {task}: \"{inst}\"")
        if len(new_entries) > 5:
            print(f"    ... and {len(new_entries) - 5} more.")

    if updated_entries:
        print("\n[~] EXISTING ENTRIES TO BE OVERWRITTEN:")
        for task, old_inst, new_inst in updated_entries[:5]:
            print(f"    {task}")
            print(f"      - Old: \"{old_inst}\"")
            print(f"      + New: \"{new_inst}\"")
        if len(updated_entries) > 5:
            print(f"    ... and {len(updated_entries) - 5} more.")

    if removed_entries:
        print("\n[-] ENTRIES TO BE REMOVED (Stale tasks not found in BDDL/Excel):")
        for task, old_inst in removed_entries[:5]:
            print(f"    {task}")
            print(f"      - Was: \"{old_inst}\"")
        if len(removed_entries) > 5:
            print(f"    ... and {len(removed_entries) - 5} more.")

    # =========================================================================
    # 4. WRITE EXECUTION GUARD
    # =========================================================================
    if DRY_RUN:
        print(f"\n[!] DRY_RUN is active. No modifications were written to disk.")
        return

    # Hard run completely overwrites the file, naturally purging removed_entries
    print(f"\n[*] Exporting to {OUTPUT_PY_PATH}...")
    write_language_map(language_map)
        
    print(f"\n[SUCCESS] Saved mapping for {len(language_map)} tasks!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-flow-root",
        help="Directory containing libero_ablation_d2 through libero_ablation_d6.",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.ablation_flow_root:
        sync_ablation_flow(args.ablation_flow_root, dry_run=not args.write)
    else:
        generate_language_map()

import os
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Execution Toggles: Choose which audits to run
RUN_INDUSTRY_TASKS = True
RUN_ABLATION_TASKS = True

# Safety Toggle: Set to True for a "Dry Run" (prints issues, changes nothing).
# Set to False for a "Hard Fix" (automatically writes fixes, e.g., JSON rewriting).
DRY_RUN = True

# ==============================================================================
# PATHS & CONSTANTS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))

BDDL_BASE = os.path.join(BASE_DIR, "libero/libero/bddl_files")
INIT_BASE = os.path.join(BASE_DIR, "libero/libero/init_files")
JSON_PATH = os.path.join(BASE_DIR, "libero/libero/benchmark/industry_env_args.json")

INDUSTRY_SUITES = [
    "libero_industry_s2d", 
    "libero_industry_d2s", 
    "libero_industry_d2d"
]

ABLATION_TARGETS = [
    "ablation_precision",
    "ablation_industry_flow"
]

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def get_basenames(directory, extension):
    """Returns a set of base filenames (without extension) from a directory."""
    if not os.path.exists(directory):
        return set()
    return {f.replace(extension, "") for f in os.listdir(directory) if f.endswith(extension)}

def get_prefix(name):
    """Extracts the first 3 parts of the name (e.g., D2D_LV1_001)."""
    parts = name.split('_')
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return name

def check_file_alignment(bddl_dir, init_dir, suite_name):
    """Checks alignment between BDDL, INIT, and PRUNED_INIT files in given directories."""
    print(f"--- Checking Suite: {suite_name.upper()} ---")
    
    bddl_names = get_basenames(bddl_dir, ".bddl")
    init_names = get_basenames(init_dir, ".init")
    pruned_names = get_basenames(init_dir, ".pruned_init")
    
    print(f"Found: {len(bddl_names)} BDDL | {len(init_names)} INIT | {len(pruned_names)} PRUNED_INIT")

    missing_init = bddl_names - init_names
    missing_pruned = bddl_names - pruned_names
    orphaned_init = init_names - bddl_names
    orphaned_pruned = pruned_names - bddl_names

    suite_issues = sum(len(x) for x in [missing_init, missing_pruned, orphaned_init, orphaned_pruned])

    if suite_issues == 0:
        print("✅ PERFECT MATCH: All files align correctly.\n")
    else:
        print("❌ MISMATCHES DETECTED:")
        if missing_init: print(f"  -> Missing .init files for: {missing_init}")
        if missing_pruned: print(f"  -> Missing .pruned_init files for: {missing_pruned}")
        if orphaned_init: print(f"  -> Orphaned .init files (No matching BDDL): {orphaned_init}")
        if orphaned_pruned: print(f"  -> Orphaned .pruned_init files (No matching BDDL): {orphaned_pruned}")
        print("\n")
        
    return suite_issues, bddl_names

# ==============================================================================
# AUDIT ROUTINES
# ==============================================================================

def audit_json_args(all_bddl_names):
    """Audits and optionally fixes keys in industry_env_args.json."""
    print(f"\n{'='*80}")
    print(f" JSON KEY VERIFICATION REPORT (industry_env_args.json vs BDDL)")
    print(f"{'='*80}\n")

    if not os.path.exists(JSON_PATH):
        print(f"❌ ERROR: JSON file not found at {JSON_PATH}\n")
        return

    with open(JSON_PATH, 'r') as f:
        try:
            env_args = json.load(f)
        except json.JSONDecodeError:
            print("❌ ERROR: Failed to parse JSON file. Check for syntax errors.\n")
            return

    bddl_prefix_map = {get_prefix(name): name for name in all_bddl_names}
    
    new_env_args = {}
    mismatches = 0
    orphans = 0

    for json_key, data in env_args.items():
        prefix = get_prefix(json_key)

        if prefix in bddl_prefix_map:
            target_bddl_name = bddl_prefix_map[prefix]
            
            if json_key != target_bddl_name:
                mismatches += 1
                print(f"  [Mismatch] JSON key: {json_key}")
                print(f"             Expected: {target_bddl_name}\n")
                
                if not DRY_RUN:
                    new_env_args[target_bddl_name] = data
                else:
                    new_env_args[json_key] = data
            else:
                new_env_args[json_key] = data
        else:
            orphans += 1
            print(f"  [Orphaned JSON Key] No BDDL found for prefix: {prefix} (Key: {json_key})\n")
            new_env_args[json_key] = data

    if mismatches == 0 and orphans == 0:
        print("✅ PERFECT MATCH: All JSON keys align correctly with BDDL filenames.\n")
    else:
        if mismatches > 0:
            if not DRY_RUN:
                with open(JSON_PATH, 'w') as f:
                    json.dump(new_env_args, f, indent=4)
                print(f"🔧 HARD FIX APPLIED: Replaced {mismatches} mismatched keys in industry_env_args.json.")
            else:
                print(f"⚠️ DRY RUN: Found {mismatches} mismatches. Set DRY_RUN = False to automatically fix them.")

def verify_industry_tasks():
    """Verifies core industry task file alignments and triggers the JSON audit."""
    print(f"\n{'='*80}")
    print(f" VERIFICATION REPORT: INDUSTRY TASKS ")
    print(f"{'='*80}\n")

    total_mismatches = 0
    global_bddl_names = set()

    for suite in INDUSTRY_SUITES:
        bddl_dir = os.path.join(BDDL_BASE, suite)
        init_dir = os.path.join(INIT_BASE, suite)
        
        issues, bddl_names = check_file_alignment(bddl_dir, init_dir, suite)
        total_mismatches += issues
        global_bddl_names.update(bddl_names)

    print(f"{'-'*80}")
    if total_mismatches == 0:
        print("🎉 SUCCESS: All core industry tasks have perfectly matching BDDL and init files!")
    else:
        print(f"⚠️ WARNING: Found {total_mismatches} discrepancies in industry files.")
    print(f"{'-'*80}\n")

    audit_json_args(global_bddl_names)

def verify_ablation_tasks():
    """Dynamically verifies ablation task file alignments."""
    grand_total = 0
    for target in ABLATION_TARGETS:
        print(f"\n{'='*80}")
        print(f" VERIFICATION REPORT: {target.upper()}")
        print(f"{'='*80}\n")

        target_bddl_base = os.path.join(BDDL_BASE, target)
        target_init_base = os.path.join(INIT_BASE, target)

        if not os.path.exists(target_bddl_base):
            print(f"❌ ERROR: BDDL base directory not found at {target_bddl_base}")
            continue

        suites = sorted([d for d in os.listdir(target_bddl_base) if os.path.isdir(os.path.join(target_bddl_base, d))])
        
        if not suites:
            print(f"⚠️ No subdirectories found in {target_bddl_base}")
            continue

        target_mismatches = 0
        for suite in suites:
            bddl_dir = os.path.join(target_bddl_base, suite)
            init_dir = os.path.join(target_init_base, suite)
            
            issues, _ = check_file_alignment(bddl_dir, init_dir, suite)
            target_mismatches += issues
            
        print(f"{'-'*80}")
        if target_mismatches == 0:
            print(f"🎉 SUCCESS: {target} has perfectly matching files!")
        else:
            print(f"⚠️ WARNING: Found {target_mismatches} discrepancies in {target}.")
        print(f"{'-'*80}\n")
        
        grand_total += target_mismatches

    return grand_total

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if DRY_RUN:
        print("\n[INFO] Running in DRY_RUN mode. No files will be modified.")
    else:
        print("\n[WARNING] Running in HARD FIX mode. Scripts will auto-correct detected issues.")

    if RUN_INDUSTRY_TASKS:
        verify_industry_tasks()
        
    if RUN_ABLATION_TASKS:
        verify_ablation_tasks()
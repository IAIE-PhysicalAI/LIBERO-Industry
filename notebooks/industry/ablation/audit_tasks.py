import os

# ==============================================================================
# audit_ablation_tasks.py
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))

# Define the specific ablation targets to audit
ABLATION_TARGETS = [
    #"ablation_precision",
    "ablation_industry_flow"
]
# ==============================================================================

def get_basenames(directory, extension):
    """Returns a set of base filenames (without extension) from a directory."""
    if not os.path.exists(directory):
        return set()
    return {f.replace(extension, "") for f in os.listdir(directory) if f.endswith(extension)}

def verify_suites(target_name):
    print(f"\n{'='*80}")
    print(f" FILE MATCH VERIFICATION REPORT: {target_name.upper()}")
    print(f"{'='*80}\n")

    bddl_base = os.path.join(BASE_DIR, f"libero/libero/bddl_files/{target_name}")
    init_base = os.path.join(BASE_DIR, f"libero/libero/init_files/{target_name}")

    if not os.path.exists(bddl_base):
        print(f"❌ ERROR: BDDL base directory not found at {bddl_base}")
        return 0

    # Dynamically find all subdirectories in the target folder
    suites = [d for d in os.listdir(bddl_base) if os.path.isdir(os.path.join(bddl_base, d))]

    if not suites:
        print(f"⚠️ No subdirectories found in {bddl_base}")
        return 0

    total_mismatches = 0

    for suite in sorted(suites):
        print(f"--- Checking Suite: {suite.upper()} ---")
        bddl_dir = os.path.join(bddl_base, suite)
        init_dir = os.path.join(init_base, suite)

        # Grab the sets of base names
        bddl_names = get_basenames(bddl_dir, ".bddl")
        init_names = get_basenames(init_dir, ".init")
        pruned_names = get_basenames(init_dir, ".pruned_init")
        
        print(f"Found: {len(bddl_names)} BDDL | {len(init_names)} INIT | {len(pruned_names)} PRUNED_INIT")

        # Find discrepancies using set differences
        missing_init = bddl_names - init_names
        missing_pruned = bddl_names - pruned_names
        orphaned_init = init_names - bddl_names
        orphaned_pruned = pruned_names - bddl_names

        suite_issues = sum(len(x) for x in [missing_init, missing_pruned, orphaned_init, orphaned_pruned])
        total_mismatches += suite_issues

        if suite_issues == 0:
            print("✅ PERFECT MATCH: All files align correctly.\n")
        else:
            print("❌ MISMATCHES DETECTED:")
            if missing_init:
                print(f"  -> Missing .init files for: {missing_init}")
            if missing_pruned:
                print(f"  -> Missing .pruned_init files for: {missing_pruned}")
            if orphaned_init:
                print(f"  -> Orphaned .init files (No matching BDDL): {orphaned_init}")
            if orphaned_pruned:
                print(f"  -> Orphaned .pruned_init files (No matching BDDL): {orphaned_pruned}")
            print("\n")

    print(f"{'-'*80}")
    if total_mismatches == 0:
        print(f"🎉 SUCCESS: {target_name} has perfectly matching BDDL, init, and pruned_init files!")
    else:
        print(f"⚠️ WARNING: Found {total_mismatches} discrepancies in {target_name}.")
    print(f"{'-'*80}\n")
    
    return total_mismatches

if __name__ == "__main__":
    grand_total_mismatches = 0
    
    for target in ABLATION_TARGETS:
        grand_total_mismatches += verify_suites(target)
        
    print(f"\n{'='*80}")
    print(f" GRAND TOTAL MISMATCHES ACROSS ALL TARGETS: {grand_total_mismatches}")
    print(f"{'='*80}\n")
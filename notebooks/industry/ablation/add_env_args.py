import os
import json
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
DRY_RUN = False  # Set to False to write the changes to the JSON file

BASE_DIR = Path(__file__).resolve().parents[3] / "libero/libero"
JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "benchmark/industry_env_args.json"))

# The three source directories to scan
BDDL_DIRS = [
    os.path.normpath(os.path.join(BASE_DIR, "bddl_files/libero_industry_ablation01a")),
    os.path.normpath(os.path.join(BASE_DIR, "bddl_files/libero_industry_ablation01b")),
    os.path.normpath(os.path.join(BASE_DIR, "bddl_files/libero_industry_ablation01c")),
    os.path.normpath(os.path.join(BASE_DIR, "bddl_files/libero_industry_ablation01d"))
]

# The strict template parameter block you provided
TEMPLATE = {
    "env_args": {
        "object_sampling": "uniform"
    },
    "set_scene": {
        "camera": {
            "azimuth": 0.0,
            "elevation": 47.16433358426285,
            "distance": 0.968694208982566
        },
        "lighting": {
            "tone": "neutral",
            "noise_level": 0.0,
            "castshadow": "false"
        },
        "table_style": {
            "color": "green",
            "finish": "matte"
        }
    },
    "takt_steps": 400
}

# ==========================================
# EXECUTION LOGIC
# ==========================================
def update_env_args():
    # 1. Load existing JSON data safely
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r') as f:
                env_args = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] The file at {JSON_PATH} contains invalid JSON.")
            return
    else:
        print(f"[WARNING] JSON not found at {JSON_PATH}. Creating a new dictionary.")
        env_args = {}

    initial_key_count = len(env_args)
    added_keys = 0

    # 2. Iterate through all specified target directories
    for directory in BDDL_DIRS:
        if not os.path.exists(directory):
            print(f"[WARNING] Directory missing, skipping: {directory}")
            continue
            
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".bddl"):
                # Isolate the task name string
                task_name = filename.replace(".bddl", "")
                
                # 3. Check existence and inject template if missing
                if task_name not in env_args:
                    # json.loads(json.dumps()) ensures a strict deep copy of the template
                    env_args[task_name] = json.loads(json.dumps(TEMPLATE))
                    added_keys += 1
                    
                    if DRY_RUN:
                        print(f"[DRY RUN] Would add key: {task_name}")

    # 4. Save modifications
    if DRY_RUN:
        print(f"\n>>> [DRY RUN] Total new entries that would be added: {added_keys} <<<")
        print(">>> Set DRY_RUN = False to execute writing sequence. <<<")
    else:
        if added_keys > 0:
            with open(JSON_PATH, 'w') as f:
                json.dump(env_args, f, indent=4)
            print(f"[SUCCESS] Saved {added_keys} new entries to {JSON_PATH}.")
            print(f"          Total keys in file: {initial_key_count + added_keys}")
        else:
            print("[INFO] No missing entries detected. JSON file is already up to date.")

if __name__ == "__main__":
    update_env_args()

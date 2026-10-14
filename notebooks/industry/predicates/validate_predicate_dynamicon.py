import os
import sys
import glob
import imageio
import numpy as np
import robosuite.utils.transform_utils as T

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.predicates.base_predicates import DynamicOn

BDDL_DIR = os.path.join(REPO_ROOT, "libero", "libero", "bddl_files")

TARGET_SUITES = [
    "libero_industry_s2d",
    "libero_industry_d2d"
]

def get_bddl_files():
    """Retrieve BDDL files from the target suites."""
    bddl_files = []
    for suite in TARGET_SUITES:
        suite_path = os.path.join(BDDL_DIR, suite)
        if os.path.exists(suite_path):
            # NEW: Changed "*.bddl" to "*_LV1_*.bddl" to strictly filter Level 1 files
            files = glob.glob(os.path.join(suite_path, "*_LV1_*.bddl"))
            bddl_files.extend(files)
        else:
            print(f"[WARNING] Suite directory not found: {suite_path}")
    return bddl_files

def validate_bddl(bddl_file, output_dir="debug_dynamicon_videos"):
    """Runs the environment for a specific BDDL and tests the DynamicOn predicate."""
    print(f"\n{'='*60}")
    print(f"Testing BDDL: {os.path.basename(bddl_file)}")
    print(f"{'='*60}")

    env_args = {
        "bddl_file_name": bddl_file,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "horizon": 200,
        "renderer": "mujoco",
        "camera_heights": 256,
        "camera_widths": 256,
        "set_scene": {
            "camera": {"azimuth": 30, "elevation": 47.7, "distance": 1},
            "lighting": {"tone": "neutral", "noise_level": 1, "castshadow": "true"},
            "table_style": {"color": "black", "finish": "matte"}
        }
    }

    try:
        env = OffScreenRenderEnv(**env_args)
        env.seed(42)
        obs = env.reset()
    except Exception as e:
        print(f"[ERROR] Failed to load environment: {e}")
        return

    # 1. Unwrap the environment to access custom BDDL properties
    base_env = env.env

    # 2. Parse the Goal State for DynamicOn Targets using the unwrapped env
    goal_states = base_env.parsed_problem.get("goal_state", [])
    dynamicon_targets = []
    for state in goal_states:
        if state[0].lower() == "dynamicon":
            obj_name = state[1]
            conveyor_name = state[2]
            dynamicon_targets.append((obj_name, conveyor_name))

    if not dynamicon_targets:
        print("[INFO] No 'DynamicOn' goal states found in this BDDL. Skipping.")
        env.close()
        return

    # 3. Get Available Dynamic Grooves/Trays from the base env
    available_grooves = getattr(base_env, 'dynamic_groove_names', []).copy()
    
    print(f"[DEBUG] Found targets to evaluate: {[t[0] for t in dynamicon_targets]}")
    print(f"[DEBUG] Available dynamic slots: {available_grooves}")

    # 4. Teleport Objects into the generated slots
    for obj_name, conv_name in dynamicon_targets:
        if not available_grooves:
            print(f"[WARNING] Not enough dynamic slots for {obj_name}")
            break
            
        g_name = available_grooves.pop(0)
        site_name = f"{g_name}_goal_region"
        
        try:
            # Extract target slot position and rotation matrix
            site_id = env.sim.model.site_name2id(site_name)
            site_pos = env.sim.data.site_xpos[site_id].copy()
            site_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
            
            # Fetch the object and its joint addresses
            obj = base_env.objects_dict[obj_name]
            joint_name = obj.joints[-1]
            joint_qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
            joint_qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
            
            # 1. FIX SUBMERGING: Calculate true drop position using the object's bottom_offset.
            # This ensures the bottom of the object (not the COM) is exactly 5cm above the tray.
            global_bottom_offset = site_mat.dot(obj.bottom_offset)
            drop_pos = site_pos - global_bottom_offset + np.array([0, 0, 0.05])
            
            # 2. FIX QUATERNION: Keep the object's natural upright/flat resting orientation
            current_quat = env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]].copy()
            
            # Apply Position Teleportation
            env.sim.data.qpos[joint_qpos_addr[0]: joint_qpos_addr[0] + 3] = drop_pos
            env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]] = current_quat
            
            # 3. FIX BOUNCING / 穿墙: Zero out angular velocity and match the conveyor's X/Y velocity.
            # This allows the object to fall perfectly vertically RELATIVE to the moving tray.
            env.sim.data.qvel[joint_qvel_addr[0]:joint_qvel_addr[1]] = 0.0 # Clear old velocities
            
            try:
                # Impart the conveyor's current speed to the falling object
                conv_speed = getattr(base_env, 'conveyor_speed', 0.05)
                forward_vec = getattr(base_env, 'forward_vec', np.array([0, 1, 0]))
                
                # Apply X and Y velocity, leave Z (index 2) at 0 to allow gravity to pull it down
                env.sim.data.qvel[joint_qvel_addr[0]:joint_qvel_addr[0]+2] = (forward_vec * conv_speed)[0:2]
            except Exception as vel_e:
                print(f"[DEBUG] Could not fetch conveyor speed: {vel_e}")

            print(f"[TELEPORT] {obj_name} -> dropped cleanly 5cm above {g_name} with matched velocity")
            
        except Exception as e:
            print(f"[ERROR] Teleport failed for {obj_name}: {e}")

    env.sim.forward()

    # 5. Simulate Forward to trigger freefall and "Sticky" physics
    frames = [obs["agentview_image"][::-1]]
    for _ in range(40): # Simulate for 2 seconds (at 20Hz)
        obs, _, _, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])

    # 6. Evaluate Predicate
    print(f"\n--- EVALUATING 'DYNAMICON' PREDICATE ---")
    dynamicon_checker = DynamicOn()
    
    class MockArg:
        def __init__(self, name, current_env):
            self.object_name = name
            self.env = current_env

    all_passed = True
    for obj_name, conv_name in dynamicon_targets:
        # Pass the base_env to the MockArg so the predicate can access env.dynamic_groove_names
        arg1 = MockArg(obj_name, base_env)
        arg2 = MockArg(conv_name, base_env)
        
        is_dynamicon = dynamicon_checker(arg1, arg2)
        status = "PASSED" if is_dynamicon else "FAILED"
        if not is_dynamicon:
            all_passed = False
        print(f"[{status}] DynamicOn({obj_name}, {conv_name})")

    # 7. Save Video
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(bddl_file).replace(".bddl", ".mp4")
    video_path = os.path.join(output_dir, video_name)
    imageio.mimsave(video_path, frames, fps=20)
    print(f"[INFO] Video saved to: {video_path}")
    
    env.close()
    return all_passed

if __name__ == "__main__":
    files = get_bddl_files()
    if not files:
        print("No BDDL files found. Exiting.")
        sys.exit(0)
        
    print(f"Found {len(files)} BDDL files. Testing the first 4...")
    # This slice extracts exactly the first 4 files
    for f in files[:4]:
        validate_bddl(f)

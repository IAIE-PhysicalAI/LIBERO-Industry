import argparse
import os
import sys
import re
import imageio
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.predicates.base_predicates import On, ReleasedOn

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def parse_bddl_goal(bddl_path):
    with open(bddl_path, 'r') as f:
        content = f.read()

    goal_match = re.search(r'\(:goal\s*\(And\s*\((.*?)\)\)', content, re.IGNORECASE | re.DOTALL)
    if not goal_match:
        raise ValueError("Could not find a valid (:goal (And (...))) block.")

    goal_str = goal_match.group(1).strip().replace('(', '').replace(')', '')
    tokens = goal_str.split()
    
    predicate = tokens[0]
    target_obj = tokens[1]
    target_placement = tokens[2]

    return predicate, target_obj, target_placement

def get_real_libero_object(env, entity_name):
    """
    Searches the environment's internal registries to find the actual 
    object wrapper containing the .check_ontop() method.
    """
    # Look in the parsed problem objects
    if hasattr(env, 'parsed_problem') and hasattr(env.parsed_problem, 'objects'):
        for obj in env.parsed_problem.objects:
            if getattr(obj, 'name', '') == entity_name or getattr(obj, 'object_name', '') == entity_name:
                return obj
                
    # Fallback to logical states objects
    if hasattr(env, 'logical_states') and hasattr(env.logical_states, 'objects'):
        for obj in env.logical_states.objects:
            if getattr(obj, 'name', '') == entity_name or getattr(obj, 'object_name', '') == entity_name:
                return obj

    # Fallback: create a runtime wrapper if the object is purely a structural site
    class NativeSiteWrapper:
        def __init__(self, name, current_env):
            self.object_name = name
            self.env = current_env
            self.object_state_type = "site"

        def check_ontop(self, other_obj):
            # Fallback topological check leveraging Robosuite's transform utils
            import robosuite.utils.transform_utils as T
            try:
                # Get Target Site Position
                site_id = self.env.sim.model.site_name2id(self.object_name)
                site_pos = self.env.sim.data.site_xpos[site_id]
                
                # Get Object Position
                obj_joint = f"{other_obj.object_name}_joint0"
                joint_addr = self.env.sim.model.get_joint_qpos_addr(obj_joint)
                obj_pos = self.env.sim.data.qpos[joint_addr[0]: joint_addr[0] + 3]
                
                # Z-height check and XY bounding box check
                z_check = obj_pos[2] >= site_pos[2]
                xy_dist = np.linalg.norm(obj_pos[:2] - site_pos[:2])
                
                return z_check and (xy_dist < 0.1)
            except Exception as e:
                print(f"[Warning] Native check_ontop failed: {e}")
                return False

    return NativeSiteWrapper(entity_name, env)

def run_native_predicate_validation(bddl_file):
    predicate, target_obj, target_placement = parse_bddl_goal(bddl_file)
    print(f"Parsed Goal -> Predicate: {predicate} | Object: {target_obj} | Target: {target_placement}")

    env_args = {
        "bddl_file_name": bddl_file,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "renderer": "mujoco",
        "control_freq": 20,
        "horizon": 100,
        "camera_heights": 512,
        "camera_widths": 512,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    obs = env.reset()

    # Retrieve placement position for teleportation
    try:
        site_id = env.sim.model.site_name2id(target_placement)
        placement_pos = env.sim.data.site_xpos[site_id].copy()
    except Exception:
        raise RuntimeError(f"Could not locate site {target_placement} in simulation.")
    
    # Teleport Object
    drop_pos = placement_pos.copy()
    drop_pos[2] += 0.05  # Reduced drop height to prevent violent bounce off stiff plate

    target_joint = f"{target_obj}_joint0"
    joint_qpos_addr = env.sim.model.get_joint_qpos_addr(target_joint)
    default_quat = np.array([1.0, 0.0, 0.0, 0.0]) 
    
    env.sim.data.qpos[joint_qpos_addr[0]: joint_qpos_addr[0] + 3] = drop_pos
    env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]] = default_quat
    env.sim.forward()

    steps = 50
    frames = [obs["agentview_image"][::-1]]
    for _ in range(steps):
        obs, _, _, _ = env.step([0.] * 7) 
        frames.append(obs["agentview_image"][::-1])

    print("\n" + "="*50)
    print("NATIVE PREDICATE EVALUATION RESULTS")
    print("="*50)

    # Instantiate both predicates to compare logic
    on_checker = On()
    ReleasedOn_checker = ReleasedOn()

    # Fetch actual objects bound to the environment
    arg1_obj = get_real_libero_object(env, target_obj)
    arg2_obj = get_real_libero_object(env, target_placement)

    # Evaluate
    is_on = on_checker(arg1_obj, arg2_obj)
    is_exact_on = ReleasedOn_checker(arg1_obj, arg2_obj)

    print(f"[{'PASS' if is_on else 'FAIL'}] On({target_obj}, {target_placement}) == {is_on}")
    print(f"[{'PASS' if is_exact_on else 'FAIL'}] ReleasedOn({target_obj}, {target_placement}) == {is_exact_on}")
    
    if is_on and not is_exact_on:
        print("\n[DIAGNOSTIC] The object is physically 'On' the target, but 'ReleasedOn' failed.")
        print("This confirms the Gripper/Contact penalty rule is rejecting the state.")

    print("="*50 + "\n")

    try:
        env.close()
    except Exception:
        pass

    video_name = os.path.join(CURRENT_DIR, "native_predicate_test.mp4")
    imageio.mimsave(video_name, frames, fps=20)
    print(f"Video saved to: {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bddl_file", help="Exact BDDL file to validate.")
    run_native_predicate_validation(os.path.abspath(parser.parse_args().bddl_file))

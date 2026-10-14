import os
import cv2
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

# --- ADD THESE TWO LINES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Configuration
SAVE_DIR = "debug_renders"
os.makedirs(SAVE_DIR, exist_ok=True)

SUITES = ["libero_industry_s2d", "libero_industry_d2s", "libero_industry_d2d"]
LEVELS = ["LV1", "LV2", "LV3"]
IMAGES_PER_SUITE_LEVEL = 3

def capture_debug_images():
    for suite_name in SUITES:
        try:
            task_suite = benchmark.get_benchmark_dict()[suite_name]()
        except KeyError:
            print(f"[SKIP] Suite {suite_name} not registered. Skipping.")
            continue
            
        for level in LEVELS:
            # 1. Filter tasks by level
            level_tasks = [
                (i, task_suite.get_task(i)) for i in range(task_suite.n_tasks) 
                if f"_{level}_" in task_suite.get_task(i).name
            ]
            
            if not level_tasks:
                continue
                
            # Randomly select 3 tasks from this level (or less if fewer exist)
            np.random.seed(42)
            sampled_tasks = np.random.choice(
                len(level_tasks), 
                min(IMAGES_PER_SUITE_LEVEL, len(level_tasks)), 
                replace=False
            )
            
            for idx in sampled_tasks:
                task_id, task = level_tasks[idx]
                task_name = task.name
                print(f"[RENDERING] {suite_name} | {level} | Task: {task_name}")
                
                # 2. Configure Environment for VISUAL OUTPUT
                env_args = {
                    "bddl_file_name": os.path.join(LIBERO_ROOT, "libero/libero/bddl_files", suite_name, os.path.basename(task.bddl_file)), # Ensure path logic matches your system
                    "camera_heights": 256,
                    "camera_widths": 256,
                    "has_renderer": False,
                    "has_offscreen_renderer": True,  # CRITICAL: Must be True to render
                    "use_camera_obs": True,          # CRITICAL: Must be True to extract image
                    "control_freq": 20,
                    "renderer": "mujoco",
                    "object_sampling": "uniform" if level == "LV1" else "random",
                    "robots": ["Panda"],
                    "gripper_types": "default"
                }

                # Note: Insert your get_scene_from_tracker(task_name) logic here 
                # if you want to test the camera/lighting randomizations too.
                
                env = None
                try:
                    env = OffScreenRenderEnv(**env_args)
                    env.seed(42 + idx)
                    
                    # Force PRNG explicitly
                    if hasattr(env.env, 'placement_initializer'):
                        env.env.placement_initializer.sample()
                        
                    obs = env.reset()
                    
                    # Settle physics
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.zeros(7))
                    
                    # 3. Extract and Save Image
                    # Check your specific camera name (usually agentview_image or robot0_eye_in_hand_image)
                    img_array = obs.get('agentview_image') 
                    
                    if img_array is not None:
                        # MuJoCo/Robosuite returns images upside down. Flip vertically.
                        img_array = np.flipud(img_array)
                        
                        # Convert RGB to BGR for OpenCV
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        file_name = f"{suite_name}_{level}_{task_name[:20]}.png"
                        cv2.imwrite(os.path.join(SAVE_DIR, file_name), img_bgr)
                    else:
                        print(f"  -> [WARNING] No 'agentview_image' found in observation.")
                        
                except Exception as e:
                    print(f"  -> [ERROR] Failed to render {task_name}: {e}")
                finally:
                    if env:
                        env.close()

if __name__ == "__main__":
    capture_debug_images()
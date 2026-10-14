import os, imageio, traceback
try:
    import pandas as pd
except ImportError:
    pd = None
import robosuite
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.task_generation_utils import generate_bddl_from_task_info
import random
import numpy as np

class TaskTracker:
    def __init__(self, excel_path, columns):
        if pd is None:
            raise RuntimeError(
                "TaskTracker requires pandas. Install pandas to run the Excel-tracker workflows."
            )
        self.excel_path = excel_path
        self.columns = columns
        if os.path.exists(excel_path):
            self.df = pd.read_excel(excel_path)
        else:
            self.df = pd.DataFrame(columns=columns)

    def log_task(self, data_dict):
        task_id = data_dict.get("Task ID")
        self.df = self.df[self.df["Task ID"] != task_id]
        self.df = pd.concat([self.df, pd.DataFrame([data_dict])], ignore_index=True)
        self.df.to_excel(self.excel_path, index=False)
        
    def print_paths(self):
        print("\n" + "="*50)
        print("✅ GENERATION COMPLETE. OUTPUT PATHS:")
        print(f"-> Excel Tracker: {self.excel_path}")
        print(f"-> BDDL Files:    {os.path.dirname(self.excel_path)}/bddl_files/")
        print(f"-> MP4 Videos:    {os.path.dirname(self.excel_path)}/videos/")
        print("="*50 + "\n")
        
def execute_and_log(
    scene_name,
    folder,
    bddl_only,
    save_video,
    vid_dir,
    task_id,
    scene_cfg,
    sampling_mode,
    tracker,
    log_data,
    debug_callback=None,
):
    bddl_files, fails = generate_bddl_from_task_info(folder=folder)
    status, error_msg, vid_file = "Fail", str(fails), ""
    
    if not fails and bddl_files:
        bddl_path = bddl_files[0]
        with open(bddl_path, 'r') as f:
            bddl_content = f.read()
            
        bddl_content = bddl_content.replace("_Ablation", "_Manipulation")
        
        with open(bddl_path, 'w') as f:
            f.write(bddl_content)

        if bddl_only:
            status, error_msg = "Success", ""
            vid_file = "BDDL_ONLY"
        else:
            try:
                robosuite_version = getattr(robosuite, "__version__", "")
                is_robosuite_1_4 = robosuite_version == "1.4.0"
                env_kwargs = {
                    "bddl_file_name": bddl_files[0],
                    "camera_heights": 256,
                    "camera_widths": 256,
                    "has_renderer": False,
                    "has_offscreen_renderer": save_video,
                    "use_camera_obs": save_video,
                    "control_freq": 20,
                    "renderer": "mujoco",
                    "set_scene": scene_cfg,
                    "object_sampling": sampling_mode,
                    "robots": ["Panda"],
                    "gripper_types": "default",
                }

                try:
                    env = OffScreenRenderEnv(**env_kwargs)
                except TypeError:
                    # Robosuite 1.4 environments may not accept the 1.5 renderer
                    # keyword. Retry only on that legacy API surface.
                    if not is_robosuite_1_4:
                        raise
                    env_kwargs.pop("renderer")
                    env = OffScreenRenderEnv(**env_kwargs)

                # Lock global RNG states because Robosuite 1.5+ relies on them.
                random.seed(task_id)
                np.random.seed(task_id)

                # Gym and Gymnasium expose seeding/reset results differently.
                seed_func = getattr(env, "seed", None)
                if callable(seed_func):
                    try:
                        env.seed(task_id)
                    except Exception:
                        pass

                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                
                # =================================================================
                # PHYSICS INTEGRITY & MESH COLLISION SCANNER (Step 0)
                # =================================================================
                # FIX 1: Append '_main' to match MuJoCo's internal physics tree compilation
                fixtures = [
                    "conveyor_belt_1_main", 
                ]
                for f_name in fixtures:
                    try:
                        f_id = env.sim.model.body_name2id(f_name)
                        # body_xmat is a flat 9-element array. Index 8 is the Z-component of the Up-Vector.
                        z_up = env.sim.data.body_xmat[f_id][8] 
                        if z_up < 0.99:
                            raise RuntimeError(f"PHYSICS FAIL: '{f_name}' is slanted (Z-up={z_up:.3f}). It clipped the robot pedestal or fell off the table!")
                    # FIX 2: Catch ValueError (Robosuite's default for missing bodies) 
                    except (KeyError, ValueError):
                        pass # Fixture not in this specific suite's scene
                        
                # 2. BROADENED CONTACT SCANNER
                for i in range(env.sim.data.ncon):
                    contact = env.sim.data.contact[i]
                    g1 = env.sim.model.geom_id2name(contact.geom1) or ""
                    g2 = env.sim.model.geom_id2name(contact.geom2) or ""
                    
                    # Broaden search to include the pedestal/mount, but ignore the table
                    rob_inv = any(t in g1.lower() or t in g2.lower() for t in ["robot", "pedestal", "mount", "base"])
                    conv_inv = "conveyor" in g1.lower() or "conveyor" in g2.lower()
                    tab_inv = "table" in g1.lower() or "table" in g2.lower()
                    
                    if rob_inv and conv_inv and not tab_inv:
                        raise RuntimeError(f"MESH COLLISION: '{g1}' physically clipping '{g2}'!")
                # =================================================================
                
                frames = []
                if save_video and "agentview_image" in obs:
                    frames.append(obs["agentview_image"][::-1])

                if is_robosuite_1_4:
                    try:
                        zero_action = np.zeros(env.action_dim)
                    except AttributeError:
                        # Legacy wrappers do not consistently expose action_dim.
                        zero_action = np.zeros(7)
                else:
                    try:
                        zero_action = np.zeros(env.action_dim)
                    except AttributeError:
                        # Keep using the native 1.5 action dimension when the
                        # LIBERO wrapper does not proxy the attribute.
                        zero_action = np.zeros(env.env.action_dim)
                    
                for _ in range(100):
                    step_result = env.step(zero_action)
                    if len(step_result) == 5:
                        obs, _, terminated, truncated, _ = step_result
                        d = bool(terminated or truncated)
                    else:
                        obs, _, d, _ = step_result
                    if save_video and "agentview_image" in obs:
                        frames.append(obs["agentview_image"][::-1])
                    if d: break

                if debug_callback is not None:
                    try:
                        debug_callback(env, log_data)
                    except Exception as error:
                        print(f"[DEBUG] Metric collection failed: {error}")
                    
                env.close()
                
                if save_video and frames:
                    vid_file = f"{scene_name}.mp4"
                    imageio.mimsave(os.path.join(vid_dir, vid_file), frames, fps=30)
                else:
                    vid_file = "SIMULATED_NO_VIDEO"
                    
                status, error_msg = "Success", ""
                
            except Exception as e:
                error_msg = f"Sim Error: {str(e)}"
                print(f"\n[!] MUJOCO CRASH ON {scene_name}:")
                print(traceback.format_exc()) # Prints the exact crash to your terminal
                
    log_data.update({"Status": status, "Video": vid_file, "Error": error_msg})
    tracker.log_task(log_data)

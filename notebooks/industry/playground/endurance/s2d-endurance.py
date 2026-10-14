import os, imageio, math, random
import numpy as np
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# ==============================================================================
# ENDURANCE SUITE CONFIGURATION
# ==============================================================================
CONFIG = {
    "task_name": "S2D_Endurance_LV2_Test",
    "bddl_folder": "endurance_bddl",
    "video_name": "endurance_test.mp4",
    "seed": 42,
    
    # --- ENDURANCE VARIABLES ---
    "num_objects": 15,
    "takt_time_sec_per_obj": 1.5,  
    "speed": 0.05,                  
    "target_key": "pump_bottle_flat",
    "init_obj": "pump_bottle_flat",
    "inlay_type": "pump_bottle_flat_groove",
    
    # --- QUICK TEST LIMITER ---
    "quick_test_steps": 50,  # Set to None to run the full thousands of frames
    
    "env_args": {
        "robots": ["Panda"],
        
        # --- GRIPPER FIX HERE ---
        # Assuming your suction_gripper is registered in robosuite as "SuctionGripper"
        "gripper_types": "SuctionGripper", 
        
        "control_freq": 20, 
        "object_sampling": "uniform", 
        "camera_heights": 256,
        "camera_widths": 256,
    },
    "set_scene": {
        "camera": {"azimuth": 45, "elevation": 45, "distance": 1.1},
        "lighting": {"tone": "neutral", "noise_level": 0.5, "castshadow": "true"},
        "table_style": {"color": "blue", "finish": "matte"}
    }
}

# --- CALCULATION FOR TOTAL FRAMES/STEPS ---
theoretical_steps = int(CONFIG["takt_time_sec_per_obj"] * CONFIG["env_args"]["control_freq"] * (CONFIG["num_objects"] + 1))

if CONFIG.get("quick_test_steps") is not None:
    TOTAL_STEPS = CONFIG["quick_test_steps"]
    print(f"⚠️ QUICK TEST MODE: Limiting simulation to {TOTAL_STEPS} steps (Theoretical was {theoretical_steps}).")
else:
    TOTAL_STEPS = theoretical_steps


@register_mu(scene_type="industry_workbench")
class Scene_S2D_Endurance(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="industry_workbench", 
            fixture_num_info={
                "industry_workbench": 1,
                "conveyor_ghost": 1,
            }, 
            object_num_info={
                CONFIG["init_obj"]: CONFIG["num_objects"],
            }
        )

    def define_regions(self):
        # 1. Ghost Conveyor Region
        X_g, Y_g = 0.10, 0.0
        self.regions.update(self.get_region_dict(
            [X_g, Y_g], "ghost_init_region", self.workspace_name, 0.001, (0,0)
        ))
        
        # 2. GRID SPAWN LOGIC (Prevents Explosions)
        # Creates a 5x6 grid with 8cm of clearance between each bottle
        start_x = -0.40  # Start far left
        start_y = -0.20  # Start near front
        spacing = 0.08   # 8cm between objects
        
        for i in range(CONFIG["num_objects"]):
            row = i // 6   # 6 objects per row
            col = i % 6
            
            x = start_x + (col * spacing)
            y = start_y + (row * spacing)
            
            # Create a tiny, exact region for each specific object
            self.regions.update(self.get_region_dict(
                [x, y], f"grid_region_{i}", self.workspace_name, 0.001, (0,0)
            ))
            
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
            ("LoadSlot", "conveyor_ghost_1", CONFIG["inlay_type"], "y", "pulse", str(CONFIG["speed"])),
        ]
        
        # Assign each of the 30 objects to its exact grid coordinate
        for i in range(CONFIG["num_objects"]):
            states.append(("On", f"{CONFIG['init_obj']}_{i+1}", f"industry_workbench_grid_region_{i}"))
            
        return states

MU_DICT[CONFIG["task_name"].lower()] = Scene_S2D_Endurance

def run():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    
    goal_states = [("Routed", f"{CONFIG['init_obj']}_{i+1}", "conveyor_ghost_1") for i in range(CONFIG["num_objects"])]
    
    register_task_info(f"endurance test: route all {CONFIG['num_objects']} bottles to the inlay", 
                       scene_name=CONFIG["task_name"], 
                       objects_of_interest=[], 
                       goal_states=goal_states)
                       
    bddl_files, fails = generate_bddl_from_task_info(folder=CONFIG["bddl_folder"])
    if fails: raise Exception(f"BDDL Generation Failed: {fails}")

    print(f"Simulating {CONFIG['task_name']}...")
    print(f"Total Objects: {CONFIG['num_objects']} | Total Simulation Steps: {TOTAL_STEPS}")
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_files[0],
        camera_heights=CONFIG["env_args"]["camera_heights"],
        camera_widths=CONFIG["env_args"]["camera_widths"],
        has_renderer=False,           
        has_offscreen_renderer=True, 
        use_camera_obs=True,
        control_freq=CONFIG["env_args"]["control_freq"],
        renderer="mujoco",
        set_scene=CONFIG["set_scene"],
        object_sampling=CONFIG["env_args"]["object_sampling"],
        robots=CONFIG["env_args"]["robots"],
        gripper_types=CONFIG["env_args"]["gripper_types"]
    )
    
    env.seed(CONFIG["seed"])
    obs = env.reset()
    frames = [obs["agentview_image"][::-1]]
    
    for step in range(TOTAL_STEPS):
        obs, _, done, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])
        
        if step > 0 and step % 500 == 0:
            print(f"  ... Simulating step {step}/{TOTAL_STEPS}")
            
        if done: 
            print(f"  ... Goal Reached early at step {step}!")
            break
            
    env.close()
    
    print("Saving video...")
    imageio.mimsave(CONFIG["video_name"], frames, fps=30)
    print(f"Success! Endurance video saved to: {CONFIG['video_name']}")

if __name__ == "__main__":
    run()
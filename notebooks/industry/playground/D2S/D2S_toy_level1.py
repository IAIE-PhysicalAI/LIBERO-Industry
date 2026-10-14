import os, imageio
import numpy as np
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

CONFIG = {
    "task_name": "D2S_mouse_belt_offset_idk",
    "bddl_folder": "debug_bddl",
    "video_name": "tmp_video.mp4",
    "steps": 150,
    "seed": 42,
    "env_args": {
        "robots": ["Panda"],
        "gripper_types": "default",
        "control_freq": 20,
        "horizon": 200,
        "object_sampling": "uniform", # Changed to uniform for baseline stabilitypython
        "camera_heights": 256,
        "camera_widths": 256,
    },
    "set_scene": {
        "camera": {"azimuth": 0, "elevation": 47.7, "distance": 1.5},
        "lighting": {"tone": "neutral", "noise_level": 1.0, "castshadow": "true"},
        "table_style": {"color": "blue", "finish": "matte"} # Use matte for industrial baseline
    }
}

@register_mu(scene_type="industry_workbench")
class Scene_D2S_mouse_belt_offset_idk(InitialSceneTemplates):
    def __init__(self):
        # 1. Update fixture to the new upright grey_plate
        super().__init__(workspace_name="industry_workbench", 
                         fixture_num_info={
            "industry_workbench": 1,
            "conveyor_belt": 1,
            "grey_plate": 1, # Switched from grey_tray
        }, 
                         object_num_info={
            "thomas_upright": 1,
            "porter_upright": 1,
            "school_bus_upright": 1,

        })

    def define_regions(self):
        # 2. Apply "Sweet Spot" Coordinates for X: [-0.20, 0.10]
        # Conveyor moves to the right edge, Plate moves to the center-left
        self.regions.update(self.get_region_dict([0.10, 0], "conveyor_init_region", self.workspace_name, 0.001, (np.pi/2, np.pi/2)))
        self.regions.update(self.get_region_dict([-0.175, 0], "plate_init_region", self.workspace_name, 0.1, (0,0)))
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        # 3. Update naming to match the upright plate sites
        return [
            ("On", "conveyor_belt_1", "industry_workbench_conveyor_init_region"),
            ("AtSpeed", "conveyor_belt_1", "-0.08", "steady"),
            ("On", "thomas_upright_1", "conveyor_belt_1_offset_region_2"),
            ("On", "mario_upright_1", "conveyor_belt_1_offset_region_1"),
            ("On", "school_bus_upright_1", "conveyor_belt_1_offset_region_3"),
            ("On", "grey_plate_1", "industry_workbench_plate_init_region"),
        ]

# Bind task name to MU_DICT for internal Libero registration
MU_DICT[CONFIG["task_name"].lower()] = Scene_D2S_mouse_belt_offset_idk

def run():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    
    # 4. Use better language instruction for D2S
    language_instruction = "pick the mice from the moving conveyor and place them onto the grey plate"
    
    register_task_info(language_instruction, 
                       scene_name=CONFIG["task_name"], 
                       objects_of_interest=[], 
                       goal_states=[
                           # 5. Use the upright plate's goal_region site
                           ("On", "thomas_upright_1", "grey_plate_1_goal_region"),
                           ("On", "porter_upright_1", "grey_plate_1_goal_region"),
                           ("On", "school_bus_upright_1", "grey_plate_1_goal_region"),

                       ])
                       
    bddl_files, fails = generate_bddl_from_task_info(folder=CONFIG["bddl_folder"])
    if fails: raise Exception(f"BDDL Generation Failed: {fails}")

    env_args = {
        "bddl_file_name": bddl_files[0],
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "use_camera_obs": True,
        "renderer": "mujoco",
        "set_scene": CONFIG["set_scene"],
        **CONFIG["env_args"]
    }
    
    env = OffScreenRenderEnv(**env_args)
    
    print(f"Simulating {CONFIG['task_name']}...")
    env.seed(CONFIG["seed"])
    obs = env.reset()
    frames = [obs["agentview_image"][::-1]]
    
    for _ in range(CONFIG["steps"]):
        obs, _, done, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])
        if done: break
        
    env.close()
    imageio.mimsave(CONFIG["video_name"], frames, fps=30)
    print(f"Success! Video saved to: {CONFIG['video_name']}")

if __name__ == "__main__":
    run()
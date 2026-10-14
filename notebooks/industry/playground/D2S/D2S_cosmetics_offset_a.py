import os, imageio
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

CONFIG = {
    "task_name": "D2S_cosmetics_offset_a",
    "bddl_folder": "debug_bddl",
    "video_name": "tmp_video.mp4",
    "steps": 150,
    "seed": 42,
    "env_args": {
        "robots": ["Panda"],
        "gripper_types": "default",
        "control_freq": 20,
        "horizon": 200,
        "object_sampling": "random",
        "camera_heights": 256,
        "camera_widths": 256,
    },
    "set_scene": {
        "camera": {"azimuth": 30, "elevation": 60, "distance": 1.3},
        "lighting": {"tone": "neutral", "noise_level": 1, "castshadow": "true", },
        "table_style": {"color": "blue", "finish": "glossy"}
    }
}

@register_mu(scene_type="industry_workbench")
class Scene_D2S_cosmetics_offset_a(InitialSceneTemplates):
    def __init__(self):
        super().__init__(workspace_name="industry_workbench", 
                         fixture_num_info={
            "industry_workbench": 1,
            "conveyor_belt": 1,
            "cosmetics_inlay_a": 1,       

        }, 
                         object_num_info={
            "pump_bottle_flat": 1,
            "serum_bottle_flat": 1,
            "cream_bottle_flat": 1,
            "cream_jar_flat": 1,
        })

    def define_regions(self):
        self.regions.update(self.get_region_dict([0.25, 0], "ghost_init_region", self.workspace_name, 0.001, (0,0)))
        self.regions.update(self.get_region_dict([-0.2, 0], "bottle_init_region", self.workspace_name, 0.2, (0,0)))
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "conveyor_belt_1", "industry_workbench_conveyor_belt_init_region"),
            ("AtSpeed", "conveyor_belt_1", "0.02", "steady"),
            ("On", "cosmetics_inlay_a_1", "industry_workbench_carrier_region"),
            # objects on conveyor belt
            ("On", "pump_bottle_flat_1", "conveyor_belt_1_offset_region_1"),
            ("On", "serum_bottle_flat_1", "conveyor_belt_1_offset_region_2"),
            ("On", "cream_bottle_flat_1", "conveyor_belt_1_offset_region_3"),
            ("On", "cream_jar_flat_1", "conveyor_belt_1_offset_region_4"),
        ]

# Bind task name to MU_DICT for internal Libero registration
MU_DICT[CONFIG["task_name"].lower()] = Scene_D2S_cosmetics_offset_a

def run():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    register_task_info("place the cosmetic products from conveyor belt to the inlay with the object", 
                       scene_name=CONFIG["task_name"], 
                       objects_of_interest=[], 
                       goal_states=[
                       ("On", "pump_bottle_flat_1", "cosmetics_inlay_a_1_goal_area_pump_bottle"),
                       ("On", "serum_bottle_flat_1", "cosmetics_inlay_a_1_goal_area_serum_bottle"),
                       ("On", "cream_bottle_flat_1", "cosmetics_inlay_a_1_goal_area_cream_bottle"),
                       ("On", "cream_jar_flat_1", "cosmetics_inlay_a_1_goal_area_cream_jar"),
                   ])
                       
    bddl_files, fails = generate_bddl_from_task_info(folder=CONFIG["bddl_folder"])
    if fails: raise Exception(f"BDDL Generation Failed: {fails}")

    # Environment setup with on-screen rendering enabled
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_files[0],
        camera_heights=CONFIG["env_args"]["camera_heights"],
        camera_widths=CONFIG["env_args"]["camera_widths"],
        has_renderer=True,           # On-screen rendering enabled
        has_offscreen_renderer=False, # Offscreen disabled
        use_camera_obs=True,
        control_freq=CONFIG["env_args"]["control_freq"],
        renderer="mujoco",
        set_scene=CONFIG["set_scene"],
        **{k: v for k, v in CONFIG["env_args"].items() if k not in ["camera_heights", "camera_widths", "control_freq"]}
    )
    
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

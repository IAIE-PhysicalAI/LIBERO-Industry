import os, imageio
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

CONFIG = {
    "task_name": "S2D_cosmetics_inlay_a",
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
        "lighting": {"tone": "neutral", "noise_level": 1, "castshadow": "true"},
        "table_style": {"color": "blue", "finish": "glossy"}
    }
}

@register_mu(scene_type="industry_workbench")
class Scene_S2D_cosmetics_inlay_a(InitialSceneTemplates):
    def __init__(self):
        super().__init__(workspace_name="industry_workbench", 
                         fixture_num_info={
            "industry_workbench": 1,
            "conveyor_ghost": 1,
            # Removed the static inlay here, because our "Repeat" predicate 
            # will generate them dynamically.
        }, 
                         object_num_info={
            "pump_bottle_upright": 1,
            "serum_bottle_upright": 1,
            "cream_bottle_upright": 1,
            "cream_jar_upright": 1,

        })

    def define_regions(self):
        self.regions.update(self.get_region_dict([0.25, 0], "ghost_init_region", self.workspace_name, 0.001, (0,0)))
        self.regions.update(self.get_region_dict([-0.2, 0], "bottle_init_region", self.workspace_name, 0.2, (0,0)))
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
            ("On", "pump_bottle_upright_1", "industry_workbench_bottle_init_region"),
            ("On", "serum_bottle_upright_1", "industry_workbench_bottle_init_region"),
            ("On", "cream_bottle_upright_1", "industry_workbench_bottle_init_region"),
            ("On", "cream_jar_upright_1", "industry_workbench_bottle_init_region"),
            ("LoadSlot",  "conveyor_ghost_1","cosmetics_inlay_a","x", "steady", "0.02"),
        ]

# Bind task name to MU_DICT for internal Libero registration
MU_DICT[CONFIG["task_name"].lower()] = Scene_S2D_cosmetics_inlay_a

def run():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    register_task_info("place everything into the moving inlay a", 
                       scene_name=CONFIG["task_name"], 
                       objects_of_interest=[], 
                       goal_states=[
                       # The BDDL parses success if it hits the Master Site of the ghost conveyor.
                       # Our custom Python step() function will handle the strict coordinate evaluation!
                        ("Routed", "pump_bottle_upright_1", "conveyor_ghost_1"),
                        ("Routed", "serum_bottle_upright_1", "conveyor_ghost_1"),
                        ("Routed", "cream_bottle_upright_1", "conveyor_ghost_1"),
                        ("Routed", "cream_jar_upright_1", "conveyor_ghost_1"),

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

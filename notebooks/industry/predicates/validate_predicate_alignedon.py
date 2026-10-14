import os
import imageio
import numpy as np
import robosuite.utils.transform_utils as T

from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.predicates.base_predicates import AlignedOn
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# Dynamically set to the directory where this script is executed
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    "task_name": "Predicate_Playground_Mario_Cream_Jar_Alignment",
    "bddl_folder": CURRENT_DIR,
    "video_name": os.path.join(CURRENT_DIR, "mario_cream_jar_alignedon.mp4"),
    "steps": 50,
    "seed": 42,
    "env_args": {
        "robots": ["Panda"],
        "gripper_types": "default",
        "control_freq": 20,
        "horizon": 200,
        "object_sampling": "uniform",
        "camera_heights": 1024,
        "camera_widths": 1024,
    },
    "set_scene": {
        "camera": {"azimuth": 0, "elevation": 47.1, "distance": 0.9},
        "lighting": {"tone": "neutral", "noise_level": 1.0, "castshadow": "true"},
        "table_style": {"color": "black", "finish": "matte"}
    }
}
@register_mu(scene_type="industry_workbench")
class Scene_Predicate_Playground(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="industry_workbench", 
            fixture_num_info={
                "industry_workbench": 1,
                "mario_flat_inlay": 1,
                "mario_flat_groove": 1,
                "cream_jar_v_groove": 2,
            }, 
            object_num_info={
                "mario_flat": 2,
                "cream_jar_upright": 2,
            }
        )

    def define_regions(self):
        # 1. Compact 2x2 fixture grid: Mario above, cream jars below.
        self.regions.update(self.get_region_dict([-0.18, 0.16], "fixture_region_1", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([-0.18, -0.16], "fixture_region_2", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.12, 0.16], "fixture_region_3", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.12, -0.16], "fixture_region_4", self.workspace_name, 0.01, (0, 0)))

        # 2. Dummy Regions for Objects (Shifted away from fixtures)
        self.regions.update(self.get_region_dict([0.32, 0.27], "obj_region_1", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.32, 0.09], "obj_region_2", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.32, -0.09], "obj_region_3", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.32, -0.27], "obj_region_4", self.workspace_name, 0.01, (0, 0)))

        # 3. Compile kwargs
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            # Initialize Fixtures
            ("On", "mario_flat_inlay_1", "industry_workbench_fixture_region_1"),
            ("On", "mario_flat_groove_1", "industry_workbench_fixture_region_2"),
            ("On", "cream_jar_v_groove_1", "industry_workbench_fixture_region_3"),
            ("On", "cream_jar_v_groove_2", "industry_workbench_fixture_region_4"),
            
            # Initialize Objects
            ("On", "mario_flat_1", "industry_workbench_obj_region_1"),
            ("On", "mario_flat_2", "industry_workbench_obj_region_2"),
            ("On", "cream_jar_upright_1", "industry_workbench_obj_region_3"),
            ("On", "cream_jar_upright_2", "industry_workbench_obj_region_4"),
        ]

MU_DICT[CONFIG["task_name"].lower()] = Scene_Predicate_Playground

def run_playground():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    
    register_task_info(
        "test AlignedOn with Mario and cream jar fixtures",
        scene_name=CONFIG["task_name"],
        objects_of_interest=[],
        goal_states=[("AlignedOn", "mario_flat_1", "mario_flat_inlay_1_goal_region")]
    )
    
    bddl_files, fails = generate_bddl_from_task_info(folder=CONFIG["bddl_folder"])
    if fails:
        raise Exception(f"BDDL Generation Failed: {fails}")

    env_args = {
        "bddl_file_name": bddl_files[0],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "renderer": "mujoco",
        "set_scene": CONFIG["set_scene"],
        **CONFIG["env_args"]
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(CONFIG["seed"])
    obs = env.reset()
    
    # 3. Define the four displayed object-fixture pairs.
    test_cases = [
        ("mario_flat_1", "mario_flat_inlay_1_goal_region"),
        ("mario_flat_2", "mario_flat_groove_1_goal_region"),
        ("cream_jar_upright_1", "cream_jar_v_groove_1_goal_region"),
        ("cream_jar_upright_2", "cream_jar_v_groove_2_goal_region"),
    ]

    # 4. Teleport objects
    for obj_name, site_name in test_cases:
        site_id = env.sim.model.site_name2id(site_name)
        site_pos = env.sim.data.site_xpos[site_id].copy()
        
        drop_pos = site_pos
        drop_pos[2] += 0.1
        
        joint_name = f"{obj_name}_joint0"
        joint_qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        
        # Visibly tilt only the lower-right cream jar by 45 degrees.
        if obj_name == "cream_jar_upright_2":
            theta = np.pi / 4 # 45 degrees
            default_quat = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
        else:
            default_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Identity (Zero rotation)
        
        env.sim.data.qpos[joint_qpos_addr[0]: joint_qpos_addr[0] + 3] = drop_pos
        env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]] = default_quat
    
    env.sim.forward()

    frames = [obs["agentview_image"][::-1]]
    for _ in range(CONFIG["steps"]):
        obs, _, _, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])

    # 5. Evaluate Predicates Computationally
    print("\n" + "="*50)
    print("PREDICATE EVALUATION RESULTS")
    print("="*50)
    
    aligned_on_checker = AlignedOn()
    results = []
    
    for obj_name, site_name in test_cases:
        class MockArg:
            def __init__(self, name, current_env):
                self.object_name = name
                self.env = current_env
            def check_contain(self, other):
                return True 
                
        arg1 = MockArg(obj_name, env)
        arg2 = MockArg(site_name, env)
        
        is_aligned = bool(aligned_on_checker(arg1, arg2))
        results.append((obj_name, site_name, is_aligned))
        status = "PASS" if is_aligned else "FAIL"
        print(f"[{status}] AlignedOn({obj_name}, {site_name}) == {is_aligned}")

    passed = sum(is_aligned for _, _, is_aligned in results)
    print("="*50)
    print(f"FINAL RESULT: {passed}/{len(results)} predicate checks passed")

    # Stage the two requested cream-jar poses for the final diagnostic frame.
    for obj_name, site_name, tilt_y, z_offset in [
        ("cream_jar_upright_1", "cream_jar_v_groove_1_goal_region", np.pi / 4, 0.035),
        ("cream_jar_upright_2", "cream_jar_v_groove_2_goal_region", 0.0, 0.045),
    ]:
        site_id = env.sim.model.site_name2id(site_name)
        pose_pos = env.sim.data.site_xpos[site_id].copy()
        pose_pos[2] += z_offset
        pose_quat = np.array([np.cos(tilt_y / 2), 0.0, np.sin(tilt_y / 2), 0.0])
        qpos_addr = env.sim.model.get_joint_qpos_addr(f"{obj_name}_joint0")
        qvel_addr = env.sim.model.get_joint_qvel_addr(f"{obj_name}_joint0")
        env.sim.data.qpos[qpos_addr[0]:qpos_addr[0] + 3] = pose_pos
        env.sim.data.qpos[qpos_addr[0] + 3:qpos_addr[1]] = pose_quat
        env.sim.data.qvel[qvel_addr[0]:qvel_addr[1]] = 0.0

    env.sim.forward()
    final_obs = env.env._get_observations(force_update=True)
    frames.append(final_obs["agentview_image"][::-1])

    env.close()
    imageio.mimsave(CONFIG["video_name"], frames, fps=30)
    print(f"\nVideo saved to: {CONFIG['video_name']}")

    # Automatically name the PNG based on the video name
    png_path = CONFIG["video_name"].replace(".mp4", "_final_frame.png")
    
    # frames[-1] grabs the very last image generated in the loop
    imageio.imwrite(png_path, frames[-1])
    
    print(f"Final frame PNG saved to: {png_path}")


if __name__ == "__main__":
    run_playground()

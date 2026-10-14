import os
import imageio
import numpy as np
import robosuite.utils.transform_utils as T

from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.predicates.base_predicates import AllowHoldOn, ReleasedOn
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# 动态获取当前脚本所在目录
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    "task_name": "Predicate_Playground_Cream_Jar_Tray_Test",
    "bddl_folder": CURRENT_DIR,
    "video_name": os.path.join(CURRENT_DIR, "cream_jar_releasedon.mp4"),
    "steps": 60, # 稍微延长时间以确保物体平稳落下
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

# ==========================================
# 1. 注册新场景 (化妆品盘 & 电子盘)
# ==========================================
@register_mu(scene_type="industry_workbench")
class Scene_Predicate_Playground_Trays(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="industry_workbench", 
            fixture_num_info={
                "industry_workbench": 1,
                "grey_plate": 1,     # <--- 替换为 Cosmetics Tray
                "electronics_tray": 1,   # <--- 替换为 Electronics Tray
            }, 
            object_num_info={
                "cream_jar_upright": 2,     # <--- 工业域内物体
            }
        )

    def define_regions(self):
        # 托盘区域 (Y轴错开)
        self.regions.update(self.get_region_dict([0.0, 0.20], "fixture_region_1", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([0.0, -0.20], "fixture_region_2", self.workspace_name, 0.01, (0, 0)))

        # 物体初始区域 (推到一旁)
        self.regions.update(self.get_region_dict([-0.3, 0.20], "obj_region_1", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([-0.3, -0.20], "obj_region_2", self.workspace_name, 0.01, (0, 0)))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "grey_plate_1", "industry_workbench_fixture_region_1"),
            ("On", "electronics_tray_1", "industry_workbench_fixture_region_2"),
            ("On", "cream_jar_upright_1", "industry_workbench_obj_region_1"),
            ("On", "cream_jar_upright_2", "industry_workbench_obj_region_2"),
        ]

MU_DICT[CONFIG["task_name"].lower()] = Scene_Predicate_Playground_Trays

# ==========================================
# 2. 运行主函数
# ==========================================
def run_playground():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    
    register_task_info(
        "test ReleasedOn with cream jar",
        scene_name=CONFIG["task_name"],
        objects_of_interest=[],
        # BDDL 初始化必须包含一个目标，否则会报错
        goal_states=[("ReleasedOn", "cream_jar_upright_1", "grey_plate_1_goal_region")]
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
    
    # -----------------------------------------------------
    # 测试用例：两个直立面霜罐分别放到两个托盘
    # -----------------------------------------------------
    test_cases = [
        ("cream_jar_upright_1", "grey_plate_1_goal_region"),
        ("cream_jar_upright_2", "electronics_tray_1_goal_region"),
    ]

    # 记录初始帧
    frames = [obs["agentview_image"][::-1]]
    
    # 将瞬移逻辑移动到仿真循环内部
    for step in range(CONFIG["steps"]):
        
        # 当步数刚好达到 12 步时，执行瞬移操作
        if step == 12:
            print(f"[Simulation] Step 12 reached. Executing delayed teleportation...")
            
            for obj_name, site_name in test_cases:
                site_id = env.sim.model.site_name2id(site_name)
                site_pos = env.sim.data.site_xpos[site_id].copy()
                
                # 将物体瞬移到托盘上方 5 厘米处
                drop_pos = site_pos
                drop_pos[2] += 0.05 
                
                joint_name = f"{obj_name}_joint0"
                joint_qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
                
                default_quat = np.array([1.0, 0.0, 0.0, 0.0]) # 保持水平
                
                env.sim.data.qpos[joint_qpos_addr[0]: joint_qpos_addr[0] + 3] = drop_pos
                env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]] = default_quat
            
            # 【极端重要】在仿真循环中途强行修改坐标后，必须调用 forward() 
            # 刷新引擎的运动学状态，否则物体会爆炸或产生物理幽灵
            env.sim.forward()

        # 执行零动作步进，推进时间
        obs, _, _, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])

    # ==========================================
    # 3. 动态导入与谓词定义
    # ==========================================
    print("\n" + "="*50)
    print("PREDICATE EVALUATION RESULTS")
    print("="*50)
    
    released_on_checker = ReleasedOn()
    allow_hold_on_checker = AllowHoldOn()
    
    # 对每个测试用例进行平行评估
    for obj_name, site_name in test_cases:
        
        # 极简的数据载体，完美契合多态谓词对参数结构的要求
        class StateWrapper:
            def __init__(self, name, core_env):
                self.object_name = name
                self.env = core_env
                
        # 核心修复：传入脱壳后的核心环境 env.env 以及完整的字符串名称
        arg1 = StateWrapper(obj_name, env.env)
        arg2 = StateWrapper(site_name, env.env)
        
        exact_res = released_on_checker(arg1, arg2)
        allow_res = allow_hold_on_checker(arg1, arg2)
        
        print(f"[{obj_name}  -->  {site_name}]")
        print(f"  ReleasedOn     (严格要求放手) : {exact_res}")
        print(f"  AllowHoldOn (允许夹爪接触) : {allow_res}\n")
    env.close()
    
    # 保存结果视频和终点截图
    imageio.mimsave(CONFIG["video_name"], frames, fps=30)
    png_path = CONFIG["video_name"].replace(".mp4", "_final_frame.png")
    imageio.imwrite(png_path, frames[-1])
    
    print("="*50)
    print(f"Video saved to: {CONFIG['video_name']}")
    print(f"Final PNG saved to: {png_path}")

if __name__ == "__main__":
    run_playground()

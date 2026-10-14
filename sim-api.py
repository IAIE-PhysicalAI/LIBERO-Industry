import os
import sys 
from pathlib import Path
from fastapi.responses import JSONResponse
# Respect an explicit rendering choice; Linux containers default to EGL while
# Windows uses MuJoCo's native WGL backend.
if os.name != "nt":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        os.environ.setdefault("EGL_DEVICE_ID", "0")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from libero.libero import set_libero_default_path
set_libero_default_path(str(REPO_ROOT / "libero" / "libero"))

import re 
import gc 
import base64
import numpy as np
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from OpenGL import GL
import traceback
import uvicorn
import argparse


class ResetRequest(BaseModel):
    initial_state: Optional[list] = None

app = FastAPI()

class GlobalEnv:
    def __init__(self):
        self.env = None
        self.use_delta_memory = None

state = GlobalEnv()

class TaskItem(BaseModel):
    benchmark_name: str
    task_id: int
    episode_idx: int = 0  # <--- ADD THIS
    img_h: int = 256
    img_w: int = 256
    set_scene: Optional[Dict] = None
    use_delta: Optional[bool] = None 

class ActionRequest(BaseModel):
    action: list

def encode_array(arr):
    # Converts a numpy array to a lightweight text string instantly
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode('utf-8')

@app.post("/init_task")
async def init_task(item: TaskItem):
    try:
        gc.collect()
        # 1. SETUP BENCHMARK AND TASK
        benchmark = get_benchmark(item.benchmark_name)(0)
        task = benchmark.get_task(item.task_id)
        
        bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        # 2. CREATE THE ENVIRONMENT (Cleaned up and merged)
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": item.img_h,
            "camera_widths": item.img_w,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "renderer": "mujoco",
            "robots": ["Panda"],
            "gripper_types": "default",
            # "set_scene": item.set_scene,
        }
        if "industry" in item.benchmark_name.lower():

            env_args["set_scene"] = item.set_scene
        
        # Close old environment if it exists to prevent memory leaks
        if state.env is not None:
            try:
                state.env.close()
            except Exception:
                pass
            state.env = None
        gc.collect()
            
        state.env = OffScreenRenderEnv(**env_args)
        
        # Only save a preference if the client explicitly asked for one
        if item.use_delta is not None:
            state.use_delta_memory = item.use_delta
            
            for robot in state.env.env.robots:
                if hasattr(robot.controller, "use_delta"):
                    robot.controller.use_delta = state.use_delta_memory
        else:
            state.use_delta_memory = None
        
        # 3. ASK THE GPU WHO IS RENDERING
        try:
            vendor = GL.glGetString(GL.GL_VENDOR)
            renderer = GL.glGetString(GL.GL_RENDERER)
            if vendor and renderer:
                print(f"🚀 [GPU CONFIRMED] Vendor: {vendor.decode()}")
                print(f"🚀 [GPU CONFIRMED] Renderer: {renderer.decode()}")
            else:
                print("⚠️ [GPU WARNING] Context created but Vendor string is empty.")
        except Exception as gl_e:
            print(f"⚠️ [GPU ERROR] Could not query GL strings: {gl_e}")

        state.env.reset()
        
        # --- AMENDED INIT STATE LOGIC BASED ON LIBERO-INDUSTRY SOURCE ---
        state_file_name = getattr(task, "init_files_file", task.init_states_file)
        init_files = benchmark.get_task_init_states(item.task_id, embodiment="panda")
        init_files_path = str(
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / state_file_name
        )

        print(f"📦 [INIT DEBUG] Path: {init_files_path}")
        print(f"📦 [INIT DEBUG] Raw init_files type: {type(init_files)}, shape: {getattr(init_files, 'shape', 'N/A')}")
        print(f"📦 [INIT DEBUG] task.problem_folder: {task.problem_folder}")
        print(f"📦 [INIT DEBUG] state_file_name: {state_file_name}")
        
        if "_add_" in state_file_name or "_level" in state_file_name:
            init_files = init_files.reshape(1, -1)
        
        safe_idx = item.episode_idx % len(init_files)
        state_to_load = init_files[safe_idx]
        print(f"\n🚀 [INIT] Loading Initial State Index: {safe_idx} (Episode {item.episode_idx})")

        # Print what the environment expects vs what the init file contains
        sim = state.env.env.sim
        print(f"📦 [ENV]  model nq (qpos): {sim.model.nq}, nv (qvel): {sim.model.nv}, total: {sim.model.nq + sim.model.nv}")
        print(f"📦 [FILE] flat array shape: {state_to_load.shape}, inferred nv from file: {len(state_to_load) - sim.model.nq} (if nq matches)")

        state.env.set_init_state(state_to_load)

        obs = state.env.env._get_observations()

        takt_limit = getattr(state.env.env, "takt_steps", None)
        if "industry" in item.benchmark_name.lower() and (
            isinstance(takt_limit, bool)
            or not isinstance(takt_limit, int)
            or takt_limit <= 0
        ):
            raise SyntaxError(
                f"Missing or invalid Industry takt_steps: {takt_limit!r}"
            )
        match = re.match(r'^([A-Z0-9_]+[A-Z0-9])_[a-z]', task.name)
        if match:
            identifier = match.group(1)  
        else:
            identifier = task.name 
        
        return {
            "status": "success", 
            "task_description": task.language, 
            "takt_limit": takt_limit,
            "task_identifier": identifier,
            "obs": {
                "agentview_image": encode_array(obs["agentview_image"]),
                "robot0_eye_in_hand_image": encode_array(obs["robot0_eye_in_hand_image"]),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                "robot0_joint_pos": obs["robot0_joint_pos"].tolist(),
                "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
                "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
            }
        }
    except Exception as e:
        print("\n=== CRITICAL ERROR IN INIT_TASK ===")
        traceback.print_exc()
        print("===================================\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(req: ActionRequest):
    try: 
        if state.use_delta_memory is not None and state.env is not None and state.env.env is not None:
            for robot in state.env.env.robots:
                if hasattr(robot.controller, "use_delta"):
                    robot.controller.use_delta = state.use_delta_memory
        print(f"[DEBUG SERVER] Robot Controller 'use_delta' mode is: {state.use_delta_memory}")

        obs, reward, done, info = state.env.step(np.array(req.action))
        
        try:
            partial_dict = state.env.env.get_partial_success()
            pcr_transfer = float(partial_dict.get("pcr_transfer", 0.0))
            pcr_slotting = float(partial_dict.get("pcr_slotting", 0.0))
        except Exception:
            pcr_transfer = 0.0
            pcr_slotting = 0.0
            
        # --- SERVER-SIDE SUCCESS TRACKING ---
        is_success = bool(info.get("success", False))
        if is_success:
            print("\n[SIM-API] >>> FULL SUCCESS DETECTED BY ENVIRONMENT <<<")
        elif pcr_slotting > 0:
            print(f"[SIM-API] Partial Success Detected: PCR={pcr_slotting:.2f}")
        # ----
        
        return {
            "obs": {
                "agentview_image": encode_array(obs["agentview_image"]),
                "robot0_eye_in_hand_image": encode_array(obs["robot0_eye_in_hand_image"]),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                "robot0_joint_pos": obs["robot0_joint_pos"].tolist(),
                "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
                "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
            },
            "reward": float(reward),
            "done": bool(done),
            "info": {
                "success": bool(info.get("success", False)),
                "takt_limit_reached": bool(info.get("takt_limit_reached", False)),
                "conveyor_speed": float(info.get("conveyor_speed", 0.0)),
                "pcr_transfer": pcr_transfer,
                "pcr_slotting": pcr_slotting,
                "ablation_phase_metrics": info.get("ablation_phase_metrics", {})
                # ===========================
            }
        }
    except Exception as e:
        print("\n=== CRITICAL ERROR IN /step ===")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
    
@app.get("/get_obs")
async def get_obs():
    if state.env is None:
        raise HTTPException(status_code=400, detail="Env not initialized.")
    obs = state.env._get_observations()
    return {
        "obs": {
            "agentview_image": encode_array(obs["agentview_image"]),
            "robot0_eye_in_hand_image": encode_array(obs["robot0_eye_in_hand_image"]),
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            "robot0_joint_pos": obs["robot0_joint_pos"].tolist(),
            "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
            "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
        }
    }

@app.post("/reset")
async def reset(req: ResetRequest = None):
    if state.env is None:
        raise HTTPException(status_code=400, detail="Env not initialized.")
        
    gc.collect()
    
    obs = state.env.reset()
    
    if req is not None and req.initial_state is not None:
        obs = state.env.set_init_state(np.array(req.initial_state))
        
    return {
        "obs": {
            "agentview_image": encode_array(obs["agentview_image"]),
            "robot0_eye_in_hand_image": encode_array(obs["robot0_eye_in_hand_image"]),
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            "robot0_joint_pos": obs["robot0_joint_pos"].tolist(),
            "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
            "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
        }
    }

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run the MuJoCo Simulation API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to.")
    args = parser.parse_args()
    
    print(f"🚀 Starting Simulation API on port {args.port}...")
    
    uvicorn.run(app, host=args.host, port=args.port)

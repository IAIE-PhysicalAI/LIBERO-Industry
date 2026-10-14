import abc
import os
import glob
import random
import re
from collections import defaultdict

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero.benchmark.libero_suite_language_map import libero_language_map

try:
    from .libero_suite_language_map import libero_language_map
except ImportError:
    libero_language_map = {}

def _inject_train_test_splits():
    """Dynamically generates identical train/test splits and injects them into RAM."""
    industry_suites = ["libero_industry_s2d", "libero_industry_d2s", "libero_industry_d2d"]
    master_train = []
    master_test = []

    rng = random.Random(42) 

    # Hold out 2 V and 2 M tasks per level while covering each signed speed
    # exactly twice per suite (12 tasks per suite -> 36 total test tasks).
    target_distribution = {
        "LV1": ["M8", "V5", "V8", "M3"],
        "LV2": ["V8", "M3", "M5", "V3"],
        "LV3": ["M5", "M8", "V3", "V5"]
    }

    for suite in industry_suites:
        if suite not in libero_task_map:
            continue

        tasks = libero_task_map[suite]
        available_tasks = {"LV1": defaultdict(list), "LV2": defaultdict(list), "LV3": defaultdict(list)}

        for t in tasks:
            match = re.search(r'_(LV[123])_.*?_(V3|V5|V8|M3|M5|M8)_', t)
            
            if match:
                lvl, spd = match.group(1), match.group(2)
                available_tasks[lvl][spd].append(t)
            else:
                # Optional: Add a temporary debug print so you know if ANY task fails to parse
                print(f"[DEBUG] Regex failed to parse task: {t}")

        test_set = []

        # Pull exactly one random task for every required cell in the matrix
        for lvl, required_speeds in target_distribution.items():
            for spd in required_speeds:
                candidates = available_tasks[lvl][spd]
                if candidates:
                    candidates.sort() # Guarantee determinism
                    chosen = rng.choice(candidates)
                    test_set.append(chosen)

        train_set = [t for t in tasks if t not in test_set]
        master_train.extend(train_set)
        master_test.extend(test_set)

    libero_task_map["libero_industry_train"] = sorted(master_train)
    libero_task_map["libero_industry_test"] = sorted(master_test)

# EXECUTE IT IMMEDIATELY!
_inject_train_test_splits()

# ---------------------------------------------------------------------------
# STANDARD LIBERO MAPPING & BASE CLASS

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x):
    clean_name = x.replace(".bddl", "")
    if clean_name in libero_language_map:   #  If it's an industry task, grab the perturbed language.
        instruction = libero_language_map[clean_name]
        return instruction

    if clean_name.startswith("ABL"):   # industry ablation : "ABLPTD000_pick_up_..." -> "pick_up_..."
        task_str = clean_name.split("_", 1)[1]
        return " ".join(task_str.split("_"))

    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_industry_s2d",
    "libero_industry_d2s",
    "libero_industry_d2d",
    "libero_industry_train",
    "libero_industry_test",
    "libero_ablation_d0",
    "libero_ablation_d1",
    "libero_ablation_d2",
    "libero_ablation_d3",
    "libero_ablation_d4",
    "libero_ablation_d5",
    "libero_ablation_d6",
    "fractional_tuning_set",
]

SUITE_HIERARCHY = {
    "libero_ablation_d": "ablation_industry_flow",
}

task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    parent_dir = "" 
    if "libero_ablation_d" in libero_suite:
        parent_dir = SUITE_HIERARCHY["libero_ablation_d"]

        
    physical_folder = os.path.join(parent_dir, libero_suite) if parent_dir else libero_suite

    for task in libero_task_map.get(libero_suite, []):
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=physical_folder,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if "ablation" in self.name or "industry" in self.name or "tuning" in self.name or self.name == "libero_90":
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i, init_states_root=None, embodiment="panda"):
        # Loading BDDL metadata and reset-only environments must not import
        # PyTorch (and its bundled OpenMP runtime). Init-state consumers pay
        # that dependency cost only when they actually load a saved array.
        from libero.libero.utils.init_state_utils import (
            load_init_state_array,
            resolve_init_state_path,
            resolve_init_states_root,
        )

        resolved_root = resolve_init_states_root(
            get_libero_path("init_states"),
            embodiment=embodiment,
            explicit_init_root=init_states_root,
        )
        init_states_path = resolve_init_state_path(
            resolved_root,
            self.tasks[i].problem_folder,
            self.tasks[i].name,
            pruned=True,
            embodiment=embodiment,
        )
        return load_init_state_array(init_states_path)

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs

class IndustryBenchmark(Benchmark):
    def _get_real_folder(self, task_name):
        if task_name.startswith("S2D_"): return "libero_industry_s2d"
        if task_name.startswith("D2S_"): return "libero_industry_d2s"
        if task_name.startswith("D2D_"): return "libero_industry_d2d"
        return self.name

    def _make_benchmark(self):
        # BYPASS the 10-task limit from the base LIBERO class completely
        raw_tasks = list(task_maps[self.name].values())
        self.n_tasks = len(raw_tasks)
        
        corrected_tasks = []
        for task in raw_tasks:
            real_folder = self._get_real_folder(task.name)
            
            # Rebuild the Task object with the correct physical problem_folder
            corrected_task = Task(
                name=task.name,
                language=task.language,
                problem=task.problem,
                problem_folder=real_folder, 
                bddl_file=task.bddl_file,
                init_states_file=task.init_states_file
            )
            corrected_tasks.append(corrected_task)
            
        self.tasks = corrected_tasks
            
        self.tasks = corrected_tasks
         

@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_INDUSTRY_S2D(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_industry_s2d"
        self._make_benchmark()

@register_benchmark
class LIBERO_INDUSTRY_D2S(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_industry_d2s"
        self._make_benchmark()


@register_benchmark
class LIBERO_INDUSTRY_D2D(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_industry_d2d"
        self._make_benchmark()

@register_benchmark
class LIBERO_INDUSTRY_TRAIN(IndustryBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_industry_train"
        self._make_benchmark()

@register_benchmark
class LIBERO_INDUSTRY_TEST(IndustryBenchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_industry_test"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D0(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d0"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D1(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d1"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D2(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d2"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D3(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d3"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D4(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d4"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_ABLATION_D5(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d5"
        self._make_benchmark()

@register_benchmark
class LIBERO_ABLATION_D6(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_ablation_d6"
        self._make_benchmark()

@register_benchmark
class FRACTIONAL_TUNING_SET(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "fractional_tuning_set"
        self._make_benchmark()

#!/usr/bin/env python3
"""
Launch script for running ScriptedSortingPolicy within the benchmark.

Strategy: Use the existing app.py flow which handles all Isaac Sim and
genie_sim imports correctly. We inject our policy by monkey-patching
TaskBenchmark BEFORE it gets used.

Usage (inside Docker):
    # Headless mode (no display needed):
    /isaac-sim/python.sh /workspace/genie_sim_RL/scripts/run_sorting_benchmark.py

    # Graphical mode (needs display, e.g. local machine):
    /isaac-sim/python.sh /workspace/genie_sim_RL/scripts/run_sorting_benchmark.py \
        --app.headless false
"""

import os
import sys

# ── Setup paths ─────────────────────────────────────────────────────
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

genie_sim_root = "/workspace/genie_sim"
genie_sim_src = os.path.join(genie_sim_root, "source")
if os.path.exists(genie_sim_src):
    sys.path.insert(0, genie_sim_src)

if not os.environ.get("SIM_REPO_ROOT"):
    os.environ["SIM_REPO_ROOT"] = genie_sim_root

# ── Set default CLI args ────────────────────────────────────────────
# These can be overridden by passing explicit --key value on the command line.
default_args = {
    "benchmark.task_name": "warehouse_g2",
    "benchmark.sub_task_name": "sorting_packages",
    "benchmark.model_arc": "pi",
    "benchmark.policy_class": "ScriptedPolicy",
    "benchmark.num_episode": "1",
    "benchmark.enable_ros": "false",
    "app.headless": "false",          # default to graphical for local
    "app.enable_ros": "false",
}
for key, value in default_args.items():
    flag = f"--{key}"
    if flag not in sys.argv:
        sys.argv.extend([flag, value])

# ── Fallback import hook: auto-mock any missing optional module ─────
# genie_sim has deep dependency trees (openai, open3d, numba, etc.)
# that aren't needed for our benchmark flow. This hook catches any
# ImportError and provides a dummy module so imports succeed.
import types
import importlib
import importlib.abc
import importlib.machinery


class _FallbackModule(types.ModuleType):
    """A module that returns dummy objects for any attribute access."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _FallbackModule(f"{self.__name__}.{name}")

    def __call__(self, *a, **kw):
        return _FallbackModule(f"{self.__name__}()")

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


class FallbackImporter(importlib.abc.MetaPathFinder):
    """Last-resort finder: if every other finder fails, provide a stub."""

    # Packages we know are NOT in the Isaac Sim env but are imported
    # transitively by genie_sim. We only mock these, not everything.
    ALLOWED = {
        "open3d", "numba", "grasp_nms", "openai", "anthropic",
        "google", "dashscope", "zhipuai",
    }

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top in self.ALLOWED and fullname not in sys.modules:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _FallbackModule(fullname)
        mod.__path__ = []
        mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod


# Install AFTER the default finders so real packages take precedence
sys.meta_path.append(FallbackImporter())
print("[Fallback] Auto-mock importer installed for optional deps")


# ── TaskBenchmark monkey-patch via import hook ─────────────────────
class TaskBenchmarkPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook that patches TaskBenchmark after it's loaded."""

    _patched = False

    def find_module(self, fullname, path=None):
        if fullname == "geniesim.benchmark.task_benchmark" and not self._patched:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Remove ourselves temporarily to avoid recursion
        sys.meta_path.remove(self)
        try:
            module = importlib.import_module(fullname)
            self._apply_patch(module)
            TaskBenchmarkPatcher._patched = True
            return module
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

    def _apply_patch(self, module):
        """Monkey-patch TaskBenchmark to use ScriptedSortingPolicy."""
        TaskBenchmark = module.TaskBenchmark

        def _create_scripted_policy(self):
            from scripted_sorting_policy import ScriptedSortingPolicy
            print("[Patch] Creating ScriptedSortingPolicy")
            self.policy = ScriptedSortingPolicy(
                task_name=self.args.task_name,
                sub_task_name=self.args.sub_task_name,
            )
            self.policy.set_data_courier(self.data_courier)

        TaskBenchmark.create_policy = _create_scripted_policy

        def _config_task_infer(self):
            self.task_mode = "infer"

        TaskBenchmark.config_task = _config_task_infer

        _original_create_env = TaskBenchmark.create_env

        # G2_STATES_4 init values for sorting_packages
        _G2_STATES_4 = {
            "body_state": [1.57, 0.0, -0.31939525311, 1.34390352404, -1.04545222194],
            "head_state": [0.0, 0.0, 0.11464],
            "init_arm": [
                0.739033, -0.717023, -1.524419, -1.537612,
                0.27811, -0.925845, -0.839257,
                -0.739033, -0.717023, 1.524419, -1.537612,
                -0.27811, -0.925845, 0.839257,
            ],
            "init_hand": [0.0, 0.0],
        }

        def _create_env_pi(self, episode_file, instance_id):
            original_arc = self.args.model_arc
            self.args.model_arc = "pi"

            # Ensure task_config has all keys BaseEnv expects
            if "sub_task_name" not in self.task_config:
                self.task_config["sub_task_name"] = self.args.sub_task_name
            if "specific_task_name" not in self.task_config:
                self.task_config["specific_task_name"] = self.args.task_name
            if "robot_cfg" not in self.task_config:
                robot_cfg = self.task_config.get("robot", {}).get(
                    "robot_cfg", "G2_omnipicker.json"
                )
                from geniesim.utils.name_utils import robot_type_mapping
                self.task_config["robot_cfg"] = robot_type_mapping(
                    robot_cfg.split(".")[0]
                )

            # Inject G2_STATES_4 into TASK_INFO_DICT if missing
            try:
                from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
                sub = self.args.sub_task_name
                if sub not in TASK_INFO_DICT:
                    TASK_INFO_DICT[sub] = {}
                if "G2_omnipicker" not in TASK_INFO_DICT[sub]:
                    TASK_INFO_DICT[sub]["G2_omnipicker"] = _G2_STATES_4
                    print(f"[Patch] Injected G2_STATES_4 into TASK_INFO_DICT[{sub}]")
            except ImportError:
                pass

            print(f"[Patch] create_env called: episode_file={episode_file}, "
                  f"instance_id={instance_id}")
            print(f"[Patch] task_config keys: {list(self.task_config.keys())}")
            if "robot" in self.task_config:
                print(f"[Patch] robot config: {self.task_config['robot']}")

            _original_create_env(self, episode_file, instance_id)
            self.args.model_arc = original_arc

            # Log scene loading results
            if hasattr(self, 'env') and self.env is not None:
                env = self.env
                print(f"[Patch] Environment created: {type(env).__name__}")
                if hasattr(env, 'usd_objects'):
                    obj_names = list(env.usd_objects.keys()) if env.usd_objects else []
                    print(f"[Patch] USD objects in scene ({len(obj_names)}): {obj_names}")
                if hasattr(env, 'object_names'):
                    print(f"[Patch] object_names: {env.object_names}")
            else:
                print("[Patch] WARNING: env is None after create_env!")

        TaskBenchmark.create_env = _create_env_pi

        # Also patch evaluate_policy to log episode info
        _original_evaluate = TaskBenchmark.evaluate_policy

        def _evaluate_debug(self, *args, **kwargs):
            print(f"[Patch] evaluate_policy called")
            if hasattr(self, 'scene_instance_ids'):
                print(f"[Patch] scene_instance_ids: {self.scene_instance_ids}")
            if hasattr(self, 'task_generator') and self.task_generator is not None:
                tg = self.task_generator
                if hasattr(tg, 'episode_files'):
                    print(f"[Patch] episode_files: {tg.episode_files}")
                if hasattr(tg, 'num_episodes'):
                    print(f"[Patch] num_episodes: {tg.num_episodes}")
            return _original_evaluate(self, *args, **kwargs)

        TaskBenchmark.evaluate_policy = _evaluate_debug

        print("[Patch] TaskBenchmark patched for ScriptedSortingPolicy + PiEnv")


sys.meta_path.insert(0, TaskBenchmarkPatcher())

# ── Now run the standard app.py ─────────────────────────────────────
# This handles ALL Isaac Sim initialization, module imports, etc.
app_dir = os.path.join(genie_sim_src, "geniesim", "app")
app_py = os.path.join(app_dir, "app.py")
print(f"[Run] Launching via {app_py}")
print(f"[Run] sys.argv = {sys.argv}")

# CWD must be the app directory so relative paths (robot_cfg/) resolve
os.chdir(app_dir)

# Kinematics_Solver uses sys.modules["__main__"].__file__ to find
# robot_cfg/ — point it at app.py so the path resolves correctly.
sys.modules["__main__"].__file__ = app_py

# Execute app.py in the current process
exec(compile(open(app_py).read(), app_py, "exec"))

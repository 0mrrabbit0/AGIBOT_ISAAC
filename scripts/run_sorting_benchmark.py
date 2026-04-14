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

        def _create_env_pi(self, episode_file, instance_id):
            original_arc = self.args.model_arc
            self.args.model_arc = "pi"
            _original_create_env(self, episode_file, instance_id)
            self.args.model_arc = original_arc

        TaskBenchmark.create_env = _create_env_pi

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

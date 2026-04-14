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

        def _spawn_cartons_from_scene_info(self, instance_id):
            """Load cartons from scene_info.json and spawn via api_core.

            Returns dict of {carton_id: world_position} for all spawned cartons.
            """
            import json as _json
            import geniesim.utils.system_utils as _su

            scene_info_path = os.path.join(
                _su.benchmark_conf_path(), "llm_task",
                self.args.sub_task_name, str(instance_id), "scene_info.json",
            )
            if not os.path.exists(scene_info_path):
                print(f"[Spawn] scene_info.json NOT found: {scene_info_path}")
                return {}

            with open(scene_info_path) as f:
                scene_info = _json.load(f)

            layout = scene_info.get("layout", {})
            print(f"[Spawn] scene_info.json has {len(layout)} objects")

            carton_positions = {}
            for obj_id, obj_data in layout.items():
                if "carton" not in obj_id:
                    continue

                usd_name = obj_data.get("usd", "")
                xyz = obj_data.get("xyz", [0, 0, 0])
                xyzw = obj_data.get("xyzw", [0, 0, 0, 1])

                # Convert xyzw quaternion to wxyz for Isaac Sim
                quat_wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

                # USD path relative to assets dir
                usd_path = f"objects/benchmark/carton/{usd_name}/Aligned.usda"

                prim_path = f"/World/Objects/{obj_id}"

                print(f"[Spawn] Adding {obj_id}: usd={usd_name} pos={xyz}")
                try:
                    self.api_core.add_usd_obj(
                        usd_path=usd_path,
                        prim_path=prim_path,
                        label_name=obj_id,
                        position=xyz,
                        rotation=quat_wxyz,
                        scale=[1.0, 1.0, 1.0],
                        object_color=[1, 1, 1],
                        object_material="general",
                        object_mass=0.5,
                    )
                    carton_positions[obj_id] = xyz
                except Exception as e:
                    print(f"[Spawn] Failed to add {obj_id}: {e}")

            import time as _time
            _time.sleep(0.5)
            return carton_positions

        def _check_carton_prims(self):
            """Check if carton prims exist in USD stage. Returns list of found prim paths."""
            found = []
            try:
                stage = self.api_core._stage
                if stage is None:
                    print("[Check] No USD stage available")
                    return found

                # Check both possible parent paths
                for parent in ["/Workspace/World/Objects", "/World/Objects"]:
                    from pxr import Usd
                    parent_prim = stage.GetPrimAtPath(parent)
                    if parent_prim.IsValid():
                        for child in parent_prim.GetChildren():
                            name = child.GetName()
                            if "carton" in name.lower():
                                found.append(str(child.GetPath()))
                print(f"[Check] Found {len(found)} carton prims: {found[:5]}...")
            except Exception as e:
                print(f"[Check] Error checking prims: {e}")
            return found

        def _query_carton_world_positions(self, prim_paths):
            """Query actual world positions of carton prims."""
            positions = {}
            for prim_path in prim_paths:
                try:
                    pos, rot = self.api_core.get_obj_world_pose(prim_path)
                    name = prim_path.split("/")[-1]
                    positions[name] = [float(pos[0]), float(pos[1]), float(pos[2])]
                    print(f"[Pos] {name}: world=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                except Exception as e:
                    print(f"[Pos] Failed to query {prim_path}: {e}")
            return positions

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

            _original_create_env(self, episode_file, instance_id)
            self.args.model_arc = original_arc

            # ── Post-creation: check/spawn cartons ──
            if hasattr(self, 'env') and self.env is not None:
                print(f"[Patch] Environment created: {type(self.env).__name__}")

                # Check if cartons loaded from scene.usda
                carton_prims = _check_carton_prims(self)
                carton_positions = {}

                if carton_prims:
                    print(f"[Patch] Cartons found in scene.usda ({len(carton_prims)})")
                    import time as _time
                    _time.sleep(0.5)  # let scene settle
                    carton_positions = _query_carton_world_positions(self, carton_prims)
                else:
                    print("[Patch] NO cartons in scene! Spawning from scene_info.json...")
                    carton_positions = _spawn_cartons_from_scene_info(self, instance_id)
                    # Re-check after spawning
                    carton_prims = _check_carton_prims(self)
                    if carton_prims:
                        import time as _time
                        _time.sleep(0.5)
                        carton_positions = _query_carton_world_positions(self, carton_prims)
                    else:
                        print("[Patch] WARNING: Cartons still not found after spawning!")

                # Pass carton positions to the policy
                if hasattr(self, 'policy') and self.policy is not None:
                    if hasattr(self.policy, 'set_carton_positions'):
                        self.policy.set_carton_positions(carton_positions)
                        print(f"[Patch] Passed {len(carton_positions)} carton positions to policy")

                # Log all USD objects
                if hasattr(self.api_core, 'usd_objects'):
                    obj_names = list(self.api_core.usd_objects.keys())
                    print(f"[Patch] All USD objects ({len(obj_names)}): {obj_names[:10]}")

                # ── Patch env.step to hold bj1-bj4 at initial values ──
                # PiEnv.step() only commands bj5 (body_joint5). Without active
                # position control, bj1-bj4 sag under gravity, tilting the
                # upper body and making the arm appear to raise toward the head.
                try:
                    from geniesim.utils.name_utils import G2_WAIST_JOINT_NAMES
                    _env = self.env
                    _orig_step = _env.step

                    # bj1-bj4 initial values from G2_STATES_4 body_state
                    _body_hold = [1.57, 0.0, -0.31939525311, 1.34390352404]
                    # Joint names: reversed list gives [idx01..idx05], take first 4
                    _body_names = list(reversed(G2_WAIST_JOINT_NAMES))[0:4]
                    _body_indices = [
                        _env.robot_joint_indices[v] for v in _body_names
                    ]

                    def _patched_step(action):
                        result = _orig_step(action)
                        # Hold bj1-bj4 at initial values each step
                        _env.api_core.set_joint_positions(
                            [float(v) for v in _body_hold],
                            joint_indices=_body_indices,
                            is_trajectory=True,
                        )
                        return result

                    _env.step = _patched_step
                    print(f"[Patch] env.step wrapped: holding bj1-bj4 at "
                          f"{_body_hold}, names={_body_names}, "
                          f"indices={_body_indices}")
                except Exception as e:
                    print(f"[Patch] WARNING: failed to wrap env.step: {e}")
            else:
                print("[Patch] WARNING: env is None after create_env!")

        TaskBenchmark.create_env = _create_env_pi

        # Also patch evaluate_policy to pass instruction to policy
        _original_evaluate = TaskBenchmark.evaluate_policy

        def _evaluate_debug(self, *args, **kwargs):
            print(f"[Patch] evaluate_policy called")
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
# Patch the source to force-disable ROS (rclpy) even if config defaults
# have enable_ros=true. The CLI args --app.enable_ros false may not
# override correctly depending on the config parser version.
import re
app_source = open(app_py).read()
# Replace the ROS conditional block: force rclpy = None
app_source = re.sub(
    r'if\s+cfg\.app\.enable_ros\s+or\s+cfg\.benchmark\.enable_ros\s*:.*?'
    r'(?=\nelse:)',
    'if False:  # PATCHED: ROS disabled\n    pass',
    app_source,
    flags=re.DOTALL,
)
print("[Run] Patched app.py source to disable ROS/rclpy")
exec(compile(app_source, app_py, "exec"))

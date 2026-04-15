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


class _StubMeta(type):
    """Metaclass that makes stub classes behave as catch-all bases."""
    def __getattr__(cls, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _StubClass
    def __instancecheck__(cls, instance):
        return True
    def __subclasscheck__(cls, subclass):
        return True


class _StubClass(metaclass=_StubMeta):
    """A class that can be inherited from (e.g. `class SimNode(Node):`)."""
    def __init__(self, *args, **kwargs):
        pass
    def __init_subclass__(cls, **kwargs):
        pass
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _StubClass
    def __call__(self, *args, **kwargs):
        return _StubClass()
    def __bool__(self):
        return False
    def __iter__(self):
        return iter([])
    def __str__(self):
        return ""
    def __int__(self):
        return 0
    def __float__(self):
        return 0.0


class _FallbackModule(types.ModuleType):
    """A module that returns _StubClass for any attribute access.

    _StubClass is both callable and inheritable, so code like
    `class SimNode(Node):` works when Node comes from a mocked module.
    """
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _StubClass

    def __call__(self, *a, **kw):
        return _StubClass()

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
        "rclpy", "rosgraph_msgs", "std_msgs", "sensor_msgs",
        "geometry_msgs", "nav_msgs", "builtin_interfaces",
        "rcl_interfaces", "action_msgs", "unique_identifier_msgs",
        "rosidl_runtime_py", "tf2_msgs", "tf2_ros",
        "cv_bridge", "toml", "curobo",
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

                for parent in ["/Workspace/Objects", "/Workspace/World/Objects", "/World/Objects"]:
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

                # ── Query robot base world z ──
                robot_base_z = 0.0
                try:
                    for robot_path in [
                        "/Workspace/Robot", "/World/Robot",
                        "/Workspace/robot", "/World/robot",
                    ]:
                        try:
                            rpos, _ = self.api_core.get_obj_world_pose(robot_path)
                            robot_base_z = float(rpos[2])
                            print(f"[Patch] Robot base z={robot_base_z:.4f} "
                                  f"(from {robot_path})")
                            break
                        except Exception:
                            continue
                except Exception as e:
                    print(f"[Patch] WARNING: robot base query failed: {e}")

                if robot_base_z < 0.1:
                    robot_base_z = 0.83  # estimated standing height
                    print(f"[Patch] Robot base z fallback: {robot_base_z}")

                # ── Query scanner & bin world positions ──
                scanner_pos = None
                bin_pos = None
                _scanner_path = "/World/background/benchmark_scanner_000"
                _bin_path = "/World/background/benchmark_material_tray_000"
                try:
                    for prim_path, label in [
                        (_scanner_path, "scanner"),
                        (_bin_path, "bin"),
                    ]:
                        pos, rot = self.api_core.get_obj_world_pose(prim_path)
                        world_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
                        print(f"[Patch] {label}: world={world_pos}")
                        if label == "scanner":
                            scanner_pos = world_pos
                        else:
                            bin_pos = world_pos
                except Exception as e:
                    print(f"[Patch] WARNING: failed to query scanner/bin: {e}")

                # ── Parse problems.json for target carton name ──
                target_carton_name = None
                try:
                    import json as _json

                    # Build candidate paths (BUG 8: fallback if
                    # benchmark_conf_path() is stubbed)
                    problems_paths = []
                    try:
                        import geniesim.utils.system_utils as _su
                        problems_paths.append(os.path.join(
                            _su.benchmark_conf_path(), "llm_task",
                            self.args.sub_task_name, str(instance_id),
                            "problems.json",
                        ))
                    except Exception:
                        pass
                    problems_paths.append(os.path.join(
                        genie_sim_root, "source", "geniesim",
                        "benchmark", "config", "llm_task",
                        self.args.sub_task_name, str(instance_id),
                        "problems.json",
                    ))

                    problems_data = None
                    for pp in problems_paths:
                        if os.path.exists(pp):
                            with open(pp) as f:
                                problems_data = _json.load(f)
                            print(f"[Patch] Loaded problems.json from {pp}")
                            break

                    if problems_data:
                        def _find_follow_target(obj):
                            """Find Follow target in problems.json.

                            Actual format:
                              {"Follow": "carton_id|[bbox]|gripper"}
                            The carton name is the first |-delimited field.
                            """
                            if isinstance(obj, dict):
                                if "Follow" in obj:
                                    val = obj["Follow"]
                                    if isinstance(val, str):
                                        return val.split("|")[0]
                                for v in obj.values():
                                    r = _find_follow_target(v)
                                    if r:
                                        return r
                            elif isinstance(obj, list):
                                for item in obj:
                                    r = _find_follow_target(item)
                                    if r:
                                        return r
                            return None

                        target_carton_name = _find_follow_target(
                            problems_data
                        )
                        if target_carton_name:
                            print(f"[Patch] Target carton from "
                                  f"problems.json: {target_carton_name}")
                except Exception as e:
                    print(f"[Patch] WARNING: problems.json parse failed: {e}")

                # ── Select target carton position ──
                target_carton_pos = None
                if target_carton_name and target_carton_name in carton_positions:
                    target_carton_pos = carton_positions[target_carton_name]
                    print(f"[Patch] Matched target: {target_carton_name} "
                          f"at {target_carton_pos}")
                elif carton_positions:
                    name, pos = next(iter(carton_positions.items()))
                    target_carton_pos = pos
                    print(f"[Patch] WARNING: fallback to {name}")

                # ── Pass scene positions + robot z to policy ──
                if hasattr(self, 'policy') and self.policy is not None:
                    # Set robot base z
                    if hasattr(self.policy, 'ROBOT_BASE'):
                        self.policy.ROBOT_BASE[2] = robot_base_z
                        print(f"[Patch] ROBOT_BASE updated: "
                              f"{self.policy.ROBOT_BASE}")

                    if (hasattr(self.policy, 'set_scene_positions')
                            and target_carton_pos is not None):
                        self.policy.set_scene_positions(
                            carton_pos=target_carton_pos,
                            scanner_pos=scanner_pos or [0.929, 0.0, 1.163],
                            bin_pos=bin_pos or [0.300, -0.917, 0.837],
                        )

                # Log all USD objects
                if hasattr(self.api_core, 'usd_objects'):
                    obj_names = list(self.api_core.usd_objects.keys())
                    print(f"[Patch] All USD objects ({len(obj_names)}): {obj_names[:10]}")

                # ── Patch env.step to hold bj1-bj4 at initial values ──
                # PiEnv.step() only commands bj5 (body_joint5). Without active
                # position control, bj1-bj4 sag under gravity, tilting the
                # upper body and making the arm appear to raise toward the head.
                # robot_joint_indices is set in reset(), so we wrap step()
                # lazily — the first call resolves the indices.
                try:
                    from geniesim.utils.name_utils import G2_WAIST_JOINT_NAMES
                    _env = self.env
                    _orig_step = _env.step

                    # body_state is [bj5,bj4,bj3,bj2,bj1] order; we need [bj1,bj2,bj3,bj4]
                    _bs = _G2_STATES_4["body_state"]
                    _body_hold = [_bs[4], _bs[3], _bs[2], _bs[1]]
                    _body_names = list(reversed(G2_WAIST_JOINT_NAMES))[0:4]
                    _body_indices_cache = []
                    _step_counter = [0]
                    _diag_carton_name = target_carton_name
                    _shared_state = {"real_gripper_world": None}

                    # Pass shared state to policy for closed-loop
                    if hasattr(self, 'policy') and self.policy is not None:
                        self.policy._shared_state = _shared_state
                        print("[Patch] Shared state attached to policy")

                    def _patched_step(action):
                        result = _orig_step(action)
                        _step_counter[0] += 1

                        # Hold bj1-bj4
                        if not _body_indices_cache:
                            if hasattr(_env, 'robot_joint_indices'):
                                _body_indices_cache.extend(
                                    _env.robot_joint_indices[v]
                                    for v in _body_names
                                )
                                print(f"[Patch] bj1-bj4 hold active: "
                                      f"indices={_body_indices_cache}")
                        if _body_indices_cache:
                            _env.api_core.set_joint_positions(
                                [float(v) for v in _body_hold],
                                joint_indices=_body_indices_cache,
                                is_trajectory=True,
                            )

                        # Query real gripper EVERY step for closed-loop
                        try:
                            rp, _ = _env.api_core.get_obj_world_pose(
                                "/genie/gripper_r_center_link"
                            )
                            _shared_state["real_gripper_world"] = [
                                float(rp[0]), float(rp[1]), float(rp[2])
                            ]
                        except Exception:
                            pass

                        # Diagnostic logging every 30 steps
                        if _step_counter[0] % 30 == 1:
                            rg = _shared_state.get("real_gripper_world")
                            if rg:
                                msg = (f"[Diag] step={_step_counter[0]}"
                                       f" real_grip=[{rg[0]:.4f},"
                                       f"{rg[1]:.4f},{rg[2]:.4f}]")
                                if _diag_carton_name:
                                    for par in ["/Workspace/Objects"]:
                                        try:
                                            cp, _ = (
                                                _env.api_core
                                                .get_obj_world_pose(
                                                    f"{par}/"
                                                    f"{_diag_carton_name}"
                                                )
                                            )
                                            import math
                                            d = math.sqrt(sum(
                                                (rg[i] - float(cp[i]))
                                                ** 2 for i in range(3)
                                            ))
                                            msg += (
                                                f" carton=[{cp[0]:.3f},"
                                                f"{cp[1]:.3f},"
                                                f"{cp[2]:.3f}]"
                                                f" dist={d:.4f}"
                                            )
                                            break
                                        except Exception:
                                            continue
                                print(msg)

                        return result

                    _env.step = _patched_step
                    print(f"[Patch] env.step wrapped (lazy): "
                          f"will hold bj1-bj4 at {_body_hold}")
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

# Execute app.py in the current process.
# Patch out rclpy hard dependency — called unconditionally at module level.
app_source = open(app_py).read()
app_source = app_source.replace(
    "rclpy = wait_rclpy()", "rclpy = None  # PATCHED: skip wait_rclpy"
)
app_source = app_source.replace(
    "rclpy.init()", "pass  # PATCHED: skip rclpy.init()"
)
print("[Run] Patched app.py source to disable rclpy")
exec(compile(app_source, app_py, "exec"))

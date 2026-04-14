#!/bin/bash
# ============================================================================
# setup_env.sh - Deploy ScriptedSortingPolicy into the genie_sim Docker env
#
# Run this INSIDE the Docker container (uwlab_train) as root:
#   bash /workspace/genie_sim_RL/scripts/setup_env.sh
#
# Or from the host:
#   docker exec uwlab_train bash /workspace/genie_sim_RL/scripts/setup_env.sh
# ============================================================================

set -e

GENIE_SIM_SRC="/workspace/genie_sim/source"
SCRIPTS_DIR="/workspace/genie_sim_RL/scripts"

echo "=== [1/3] Installing missing Python packages ==="
# Isaac Sim bundles numpy 1.26.x; we must NOT upgrade it.
# Only install packages that are missing.

pip_install() {
    /isaac-sim/kit/python/bin/python3 -m pip install "$@" 2>&1 | tail -3
}

# shapely: needed by solver_2d
pip_install shapely 2>/dev/null || true

# scikit-learn: needed by fix_rotation.py (requires joblib, threadpoolctl)
pip_install scikit-learn 2>/dev/null || true

# future: needed by hook_base.py
pip_install future 2>/dev/null || true

# Restore numpy if it got upgraded (critical for Isaac Sim ABI compatibility)
NUMPY_VER=$(/isaac-sim/kit/python/bin/python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
echo "Current numpy version: $NUMPY_VER"
if [[ "$NUMPY_VER" != 1.26.* ]]; then
    echo "WARNING: numpy was upgraded, restoring 1.26.0..."
    pip_install 'numpy==1.26.0' --force-reinstall
fi

echo ""
echo "=== [2/3] Patching genie_sim source files for optional deps ==="

# --- transform_utils.py: wrap open3d, numba, sklearn in try/except ---
FILE="$GENIE_SIM_SRC/geniesim/utils/transform_utils.py"
if [ -f "$FILE" ] && ! grep -q "# PATCHED_OPTIONAL_DEPS" "$FILE"; then
    cp "$FILE" "$FILE.bak"
    /isaac-sim/kit/python/bin/python3 -c "
import re
with open('$FILE', 'r') as f:
    content = f.read()

# Wrap 'import open3d' in try/except
content = re.sub(
    r'^(import open3d.*)',
    r'try:\\n    \\1\\nexcept ImportError:\\n    open3d = None',
    content, flags=re.MULTILINE)

# Wrap 'import numba' in try/except
content = re.sub(
    r'^(import numba.*)',
    r'try:\\n    \\1\\nexcept ImportError:\\n    numba = None',
    content, flags=re.MULTILINE)

# Wrap 'from numba import' in try/except
content = re.sub(
    r'^(from numba import.*)',
    r'try:\\n    \\1\\nexcept ImportError:\\n    pass',
    content, flags=re.MULTILINE)

content += '\\n# PATCHED_OPTIONAL_DEPS\\n'
with open('$FILE', 'w') as f:
    f.write(content)
print('  Patched: transform_utils.py')
"
else
    echo "  transform_utils.py: already patched or not found"
fi

# --- object.py: wrap grasp_nms in try/except ---
FILE="$GENIE_SIM_SRC/geniesim/utils/object.py"
if [ -f "$FILE" ] && ! grep -q "# PATCHED_OPTIONAL_DEPS" "$FILE"; then
    cp "$FILE" "$FILE.bak"
    /isaac-sim/kit/python/bin/python3 -c "
import re
with open('$FILE', 'r') as f:
    content = f.read()

content = re.sub(
    r'^(from grasp_nms.*)',
    r'try:\\n    \\1\\nexcept ImportError:\\n    nms_grasp = lambda *a, **kw: []',
    content, flags=re.MULTILINE)

content += '\\n# PATCHED_OPTIONAL_DEPS\\n'
with open('$FILE', 'w') as f:
    f.write(content)
print('  Patched: object.py')
"
else
    echo "  object.py: already patched or not found"
fi

# --- sdf.py: wrap open3d in try/except ---
FILE="$GENIE_SIM_SRC/geniesim/plugins/tgs/layout/utils/sdf.py"
if [ -f "$FILE" ] && ! grep -q "# PATCHED_OPTIONAL_DEPS" "$FILE"; then
    cp "$FILE" "$FILE.bak"
    /isaac-sim/kit/python/bin/python3 -c "
import re
with open('$FILE', 'r') as f:
    content = f.read()

content = re.sub(
    r'^(import open3d.*)',
    r'try:\\n    \\1\\nexcept ImportError:\\n    open3d = None',
    content, flags=re.MULTILINE)

content += '\\n# PATCHED_OPTIONAL_DEPS\\n'
with open('$FILE', 'w') as f:
    f.write(content)
print('  Patched: sdf.py')
"
else
    echo "  sdf.py: already patched or not found"
fi

echo ""
echo "=== [3/4] Patching config.yaml for sorting_packages ==="
CONFIG_YAML="$GENIE_SIM_SRC/geniesim/config/config.yaml"
if [ -f "$CONFIG_YAML" ]; then
    # Add sub_task_name if missing
    if ! grep -q "sub_task_name:" "$CONFIG_YAML"; then
        # Insert after task_name line in benchmark section
        sed -i '/^  task_name:/a\  sub_task_name: "sorting_packages"' "$CONFIG_YAML"
        echo "  Added sub_task_name: sorting_packages"
    else
        echo "  sub_task_name already present"
    fi

    # Fix task_name if it's still the default gm_task_pickplace
    if grep -q 'task_name:.*gm_task_pickplace' "$CONFIG_YAML"; then
        sed -i 's/task_name:.*gm_task_pickplace.*/task_name: "warehouse_g2"/' "$CONFIG_YAML"
        echo "  Fixed task_name: warehouse_g2"
    else
        echo "  task_name already correct"
    fi
else
    echo "  WARNING: config.yaml not found at $CONFIG_YAML"
fi

echo ""
echo "=== [4/4] Verifying setup ==="
echo "Scripts dir: $SCRIPTS_DIR"
ls -la "$SCRIPTS_DIR/run_sorting_benchmark.py" "$SCRIPTS_DIR/scripted_sorting_policy.py" 2>/dev/null

echo ""
echo "============================================="
echo "  Setup complete!"
echo ""
echo "  To run the benchmark:"
echo "    # Graphical (local machine with display):"
echo "    /isaac-sim/python.sh $SCRIPTS_DIR/run_sorting_benchmark.py"
echo ""
echo "    # Headless (remote server, no display):"
echo "    /isaac-sim/python.sh $SCRIPTS_DIR/run_sorting_benchmark.py --app.headless true"
echo "============================================="

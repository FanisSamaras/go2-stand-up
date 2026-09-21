import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from gym_quadruped import robot_cfgs


# ============================================================
# Configuration
# ============================================================

cfg = robot_cfgs.get_robot_config("go2")

MODEL_PATH = cfg.mjcf_filename
MODEL_PATH = "./.venv/lib/python3.14/site-packages/gym_quadruped/robot_model/go2/go2.xml"
# MODEL_PATH = "src/models/go2"

LEG_ORDER = ["FL", "FR", "RL", "RR"]

JOINT_NAMES = []

for leg in LEG_ORDER:
    JOINT_NAMES.extend(cfg.leg_joints[leg])

FOOT_NAMES = cfg.feet_geom_names


# ============================================================
# Load MuJoCo
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# Joint information
# ============================================================

joint_info = []

for name in JOINT_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    qpos_addr = model.jnt_qposadr[jid]

    if model.jnt_limited[jid]:
        qmin, qmax = model.jnt_range[jid]
    else:
        qmin, qmax = -np.pi, np.pi

    joint_info.append(
        {
            "name": name,
            "id": jid,
            "qpos_addr": qpos_addr,
            "qmin": qmin,
            "qmax": qmax,
        }
    )


# ============================================================
# Foot geom IDs
# ============================================================

foot_geom_ids = {}

for leg, geom_name in FOOT_NAMES.items():

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if gid < 0:
        raise RuntimeError(
            f"Foot geom not found: {geom_name}"
        )

    foot_geom_ids[leg] = gid


# ============================================================
# Initial pose
# ============================================================

home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

mujoco.mj_resetDataKeyframe(
    model,
    data,
    home_id
)

initial_qpos = data.qpos.copy()
initial_qvel = data.qvel.copy()


# ============================================================
# Kinematics
# ============================================================

def forward():

    mujoco.mj_forward(model, data)


def get_feet():

    feet = {}

    for leg in LEG_ORDER:

        gid = foot_geom_ids[leg]

        feet[leg] = data.geom_xpos[gid].copy()

    return feet


def get_com():

    return data.subtree_com[0].copy()


def line_distance(point, a, b):

    p = point[:2]
    a = a[:2]
    b = b[:2]

    ab = b - a

    n = np.linalg.norm(ab)

    if n < 1e-8:
        return 0.0

    return abs(
        ab[0] * (a[1] - p[1])
        - (a[0] - p[0]) * ab[1]
    ) / n


def metrics():

    feet = get_feet()
    com = get_com()

    distance = line_distance(
        com,
        feet["FL"],
        feet["RR"],
    )

    return feet, com, distance


# ============================================================
# Set joint
# ============================================================

def set_joint(i, angle):

    info = joint_info[i]

    data.qpos[info["qpos_addr"]] = angle

    forward()


# ============================================================
# Reset
# ============================================================

def reset():

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    forward()

    for i, slider in enumerate(sliders):

        q = data.qpos[
            joint_info[i]["qpos_addr"]
        ]

        slider.set_val(
            np.degrees(q),
            emit=False,
        )


# ============================================================
# Print pose
# ============================================================

def print_pose(event=None):

    forward()

    print("\n" + "=" * 70)
    print("CURRENT POSE")
    print("=" * 70)

    q_values = []

    for info in joint_info:

        q = data.qpos[
            info["qpos_addr"]
        ]

        q_values.append(q)

        print(
            f"{info['name']:20s}: "
            f"{q:+.6f} rad "
            f"({np.degrees(q):+.2f} deg)"
        )

    print("\nnominal_q = np.array([")

    for i, q in enumerate(q_values):

        comma = "," if i < len(q_values) - 1 else ""

        print(
            f"    {q:.8f}{comma}"
        )

    print("])")

    feet, com, distance = metrics()

    print("\nFeet:")

    for leg in LEG_ORDER:

        p = feet[leg]

        print(
            f"{leg}: "
            f"x={p[0]:+.4f}, "
            f"y={p[1]:+.4f}, "
            f"z={p[2]:+.4f}"
        )

    print(
        f"\nCOM: "
        f"{com[0]:+.4f}, "
        f"{com[1]:+.4f}, "
        f"{com[2]:+.4f}"
    )

    print(
        f"\nCOM → FL/RR support line: "
        f"{distance:.4f} m"
    )

    print("=" * 70)


# ============================================================
# Matplotlib UI
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

fig.suptitle(
    "Unitree Go2 MuJoCo Pose Editor",
    fontsize=16,
)


# ------------------------------------------------------------
# Slider layout
# ------------------------------------------------------------

sliders = []

slider_height = 0.035

start_y = 0.90

for i, info in enumerate(joint_info):

    y = start_y - i * 0.055

    ax = fig.add_axes(
        [0.25, y, 0.65, slider_height]
    )

    q = data.qpos[
        info["qpos_addr"]
    ]

    slider = Slider(
        ax,
        info["name"],
        np.degrees(info["qmin"]),
        np.degrees(info["qmax"]),
        valinit=np.degrees(q),
        valstep=0.1,
    )

    sliders.append(slider)


# ============================================================
# Buttons
# ============================================================

reset_ax = fig.add_axes(
    [0.05, 0.08, 0.12, 0.05]
)

print_ax = fig.add_axes(
    [0.20, 0.08, 0.12, 0.05]
)

reset_button = Button(
    reset_ax,
    "Reset",
)

print_button = Button(
    print_ax,
    "Print Pose",
)


# ============================================================
# Slider callback
# ============================================================

def slider_changed(value):

    # Update corresponding MuJoCo joint
    for i, slider in enumerate(sliders):

        angle_rad = np.radians(
            slider.val
        )

        addr = joint_info[i]["qpos_addr"]

        data.qpos[addr] = angle_rad

    forward()


for slider in sliders:

    slider.on_changed(
        slider_changed
    )


reset_button.on_clicked(
    reset
)

print_button.on_clicked(
    print_pose
)


# ============================================================
# MuJoCo viewer
# ============================================================

viewer = mujoco.viewer.launch_passive(
    model,
    data,
)


# ============================================================
# Main UI loop
# ============================================================

try:

    while plt.fignum_exists(fig.number):

        forward()

        viewer.sync()

        plt.pause(0.01)

finally:

    viewer.close()
    plt.close(fig)

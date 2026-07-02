"""rollout_navdp_policy.py - run a NavDP/S2DiT policy in the Mars HabitatSim scene.

This is a thin adapter between this repository's Mars terrain renderer and the
NavDP route-belief policy code.  The Mars scene does not provide semantic goal
or obstacle masks, so the script creates a synthetic goal mask by projecting a
world-space target point into the camera image.  Obstacle masks are optional and
can be kept empty for first-pass target seeking.

Typical usage:

    python rollout_navdp_policy.py \
      --navdp-root /path/to/navdp_sam \
      --ckpt /path/to/navdp_sam/runs/.../ckpt_last.pt \
      --goal-x 8 --goal-z -8 \
      --out mars_navdp_rollout
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

import habitat_sim
from habitat_sim.agent import AgentConfiguration
import quaternion


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "marsyard2022_tri.glb"
DEFAULT_OBJ = HERE / "marsyard2022.obj"

SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566


class TerrainHeight:
    def __init__(
        self,
        *,
        mode: str,
        heightmap: Optional[Path],
        obj: Optional[Path],
        flat_y: float,
        size_x: float,
        size_z: float,
        size_y: float,
        flip_x: bool,
        flip_z: bool,
        swap_xz: bool,
    ):
        self.mode = mode
        self.flat_y = float(flat_y)
        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.size_y = float(size_y)
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)
        self.height = None
        self.hm_h = 0
        self.hm_w = 0
        self.obj_xs = None
        self.obj_zs = None
        self.obj_h = None

        if mode == "auto":
            if heightmap is not None and heightmap.exists():
                mode = "heightmap"
            elif obj is not None and obj.exists():
                mode = "obj"
            else:
                mode = "flat"
        self.mode = mode

        if self.mode == "heightmap":
            if heightmap is None or not heightmap.exists():
                raise FileNotFoundError(f"heightmap not found: {heightmap}")
            self._load_heightmap(heightmap)
        elif self.mode == "obj":
            if obj is None or not obj.exists():
                raise FileNotFoundError(f"OBJ terrain not found: {obj}")
            self._load_obj_grid(obj)
        elif self.mode == "flat":
            pass
        else:
            raise ValueError(f"unknown terrain height mode: {self.mode}")

    def _load_heightmap(self, path: Path) -> None:
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
        y = arr * self.size_y
        y = y - float(np.mean(y))
        self.height = y.astype(np.float32)
        self.hm_h, self.hm_w = self.height.shape

    def _load_obj_grid(self, path: Path) -> None:
        verts = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    # hm2obj.py wrote OBJ as v x row_axis height.  Blender/Habitat
                    # turns this into x/z ground plane with y-up height.
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
        if not verts:
            raise RuntimeError(f"no OBJ vertices found in {path}")
        arr = np.asarray(verts, dtype=np.float32)
        xs = np.unique(arr[:, 0])
        zs = np.unique(arr[:, 1])
        xs.sort()
        zs.sort()
        grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
        x_to_i = {float(x): i for i, x in enumerate(xs.tolist())}
        z_to_i = {float(z): i for i, z in enumerate(zs.tolist())}
        for x, z, h in arr:
            grid[z_to_i[float(z)], x_to_i[float(x)]] = h
        if np.isnan(grid).any():
            fill = float(np.nanmean(grid))
            grid = np.nan_to_num(grid, nan=fill)
        self.obj_xs = xs.astype(np.float32)
        self.obj_zs = zs.astype(np.float32)
        self.obj_h = grid.astype(np.float32)

    def __call__(self, x: float, z: float) -> float:
        if self.mode == "flat":
            return self.flat_y
        if self.mode == "heightmap":
            return self._sample_heightmap(x, z)
        return self._sample_obj(x, z)

    def _map_xz(self, x: float, z: float) -> Tuple[float, float]:
        if self.swap_xz:
            x, z = z, x
        u = (x + self.size_x / 2.0) / self.size_x
        v = (z + self.size_z / 2.0) / self.size_z
        if self.flip_x:
            u = 1.0 - u
        if self.flip_z:
            v = 1.0 - v
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def _sample_heightmap(self, x: float, z: float) -> float:
        assert self.height is not None
        u, v = self._map_xz(x, z)
        px = u * (self.hm_w - 1)
        py = v * (self.hm_h - 1)
        return bilinear_grid(self.height, px, py)

    def _sample_obj(self, x: float, z: float) -> float:
        assert self.obj_xs is not None and self.obj_zs is not None and self.obj_h is not None
        xx = float(np.clip(x, float(self.obj_xs[0]), float(self.obj_xs[-1])))
        zz = float(np.clip(z, float(self.obj_zs[0]), float(self.obj_zs[-1])))
        col = np.searchsorted(self.obj_xs, xx) - 1
        row = np.searchsorted(self.obj_zs, zz) - 1
        col = int(np.clip(col, 0, len(self.obj_xs) - 2))
        row = int(np.clip(row, 0, len(self.obj_zs) - 2))
        x0, x1 = float(self.obj_xs[col]), float(self.obj_xs[col + 1])
        z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
        tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
        tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
        h00 = float(self.obj_h[row, col])
        h10 = float(self.obj_h[row, col + 1])
        h01 = float(self.obj_h[row + 1, col])
        h11 = float(self.obj_h[row + 1, col + 1])
        h0 = h00 * (1.0 - tx) + h10 * tx
        h1 = h01 * (1.0 - tx) + h11 * tx
        return float(h0 * (1.0 - tz) + h1 * tz)


def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
    h, w = grid.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = float(px - x0)
    dy = float(py - y0)
    h00 = float(grid[y0, x0])
    h10 = float(grid[y0, x1])
    h01 = float(grid[y1, x0])
    h11 = float(grid[y1, x1])
    h0 = h00 * (1.0 - dx) + h10 * dx
    h1 = h01 * (1.0 - dx) + h11 * dx
    return float(h0 * (1.0 - dy) + h1 * dy)


def add_navdp_to_path(navdp_root: Path) -> None:
    root = navdp_root.expanduser().resolve()
    scripts = root / "scripts"
    for p in (root, scripts):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def resolve_navdp_root(raw: Optional[str]) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw))
    env = os.environ.get("NAVDP_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        HERE.parent / "navdp_sam",
        HERE.parent / "New code",
        HERE.parent / "ICRA2027" / "New code",
    ])
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "model_s2_dit.py").exists() and (c / "scripts" / "rollout_habitat_policy.py").exists():
            return c
    raise FileNotFoundError(
        "Could not find NavDP repo. Pass --navdp-root /path/to/navdp_sam "
        "or set NAVDP_ROOT."
    )


def make_sensor(uuid: str, sensor_type, height: int, width: int, hfov_deg: float):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [int(height), int(width)]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = float(hfov_deg)
    return spec


def make_sim(scene: Path, height: int, width: int, hfov_deg: float):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene.expanduser().resolve())
    sim_cfg.enable_physics = False
    rgb = make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg)
    depth = make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg)
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.asarray([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


def set_agent_pose(agent, x: float, y: float, z: float, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray([x, y, z], dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
    agent.set_state(state)


def rgb_depth(obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(obs["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def camera_coords(point: np.ndarray, position: np.ndarray, yaw: float) -> Tuple[float, float, float]:
    d = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    forward = float(fwd_x * d[0] + fwd_z * d[2])
    left = float(left_x * d[0] + left_z * d[2])
    right = -left
    up = float(d[1])
    return right, up, forward


def intrinsics_from_hfov(height: int, width: int, hfov_deg: float) -> Dict[str, float]:
    hfov = math.radians(float(hfov_deg))
    fx = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    fy = fx
    return {"fx": fx, "fy": fy, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


def draw_circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - float(u)) ** 2 + (yy - float(v)) ** 2 <= float(radius) ** 2
    return mask.astype(np.uint8)


def project_goal_mask(
    *,
    goal: np.ndarray,
    position: np.ndarray,
    yaw: float,
    height: int,
    width: int,
    hfov_deg: float,
    radius: int,
    clamp_to_edge: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    intr = intrinsics_from_hfov(height, width, hfov_deg)
    right, up, forward = camera_coords(goal, position, yaw)
    visible = forward > 0.05
    if not visible:
        return np.zeros((height, width), dtype=np.uint8), {
            "visible": 0.0,
            "u": -1.0,
            "v": -1.0,
            "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
            "bearing": float(math.atan2(right, forward if abs(forward) > 1e-6 else 1e-6)),
        }
    u = intr["cx"] + intr["fx"] * right / max(forward, 1e-6)
    v = intr["cy"] - intr["fy"] * up / max(forward, 1e-6)
    in_frame = radius <= u < width - radius and radius <= v < height - radius
    if not in_frame and clamp_to_edge:
        u = float(np.clip(u, radius, width - radius - 1))
        v = float(np.clip(v, radius, height - radius - 1))
        in_frame = True
    if not in_frame:
        return np.zeros((height, width), dtype=np.uint8), {
            "visible": 0.0,
            "u": float(u),
            "v": float(v),
            "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
            "bearing": float(math.atan2(right, forward)),
        }
    mask = draw_circle_mask(height, width, u, v, radius)
    return mask, {
        "visible": 1.0,
        "u": float(u),
        "v": float(v),
        "range": float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]])),
        "bearing": float(math.atan2(right, forward)),
    }


def depth_obstacle_mask(depth: np.ndarray, threshold: float, min_y_frac: float) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    h, _ = arr.shape
    yy = np.arange(h)[:, None]
    mask = np.isfinite(arr) & (arr > 0.0) & (arr < float(threshold)) & (yy >= h * float(min_y_frac))
    return mask.astype(np.uint8)


def overlay_frame(rgb: np.ndarray, goal_mask: np.ndarray, obstacle_mask: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pix = np.asarray(overlay).copy()
    gm = np.asarray(goal_mask) > 0
    om = np.asarray(obstacle_mask) > 0
    pix[gm] = [0, 255, 0, 120]
    pix[om] = [255, 0, 0, 100]
    overlay = Image.fromarray(pix, mode="RGBA")
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 46], fill=(0, 0, 0, 170))
    draw.text((8, 6), text, fill=(255, 255, 255, 255))
    return img.convert("RGB")


def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio unavailable; skipping video: {exc}", flush=True)
        return
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=float(fps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a trained NavDP/S2DiT policy inside the Mars HabitatSim terrain.")
    ap.add_argument("--navdp-root", default=None, help="Path to the navdp_sam repo containing model_s2_dit.py")
    ap.add_argument("--ckpt", required=True, help="Path to trained NavDP/S2DiT checkpoint")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--out", default="mars_navdp_rollout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=["model", "ema"], default="model")
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--hfov-deg", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--stop-dist", type=float, default=1.0)
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=8.0)
    ap.add_argument("--start-yaw-deg", type=float, default=0.0)
    ap.add_argument("--goal-x", type=float, required=True)
    ap.add_argument("--goal-z", type=float, required=True)
    ap.add_argument("--goal-y", type=float, default=None, help="World Y of goal marker; default terrain height + goal-height")
    ap.add_argument("--goal-height", type=float, default=0.6, help="Goal marker height above terrain when --goal-y is omitted")
    ap.add_argument("--goal-radius", type=int, default=18)
    ap.add_argument("--no-clamp-goal-to-edge", action="store_true")
    ap.add_argument("--terrain-height-mode", choices=["auto", "heightmap", "obj", "flat"], default="auto")
    ap.add_argument("--heightmap", default=None)
    ap.add_argument("--terrain-obj", default=str(DEFAULT_OBJ))
    ap.add_argument("--flat-y", type=float, default=0.0)
    ap.add_argument("--clearance", type=float, default=0.9)
    ap.add_argument("--size-x", type=float, default=SIZE_X)
    ap.add_argument("--size-z", type=float, default=SIZE_Z)
    ap.add_argument("--size-y", type=float, default=SIZE_Y)
    ap.add_argument("--flip-heightmap-x", action="store_true")
    ap.add_argument("--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--swap-heightmap-xz", action="store_true")
    ap.add_argument("--habitat-proprio-mode", choices=["pose7", "planar3", "zero"], default=None)
    ap.add_argument("--habitat-action-mode", choices=["action3d", "action2d", "waypoint"], default=None)
    ap.add_argument("--habitat-yaw-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument("--habitat-use-obstacle-channel", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--obstacle-mode", choices=["none", "depth"], default="none")
    ap.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
    ap.add_argument("--obstacle-min-y-frac", type=float, default=0.45)
    ap.add_argument("--zero-lateral", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-forward-speed", type=float, default=1.0)
    ap.add_argument("--max-lateral-speed", type=float, default=1.0)
    ap.add_argument("--max-yaw-rate", type=float, default=1.0)
    ap.add_argument("--action-smoothing", choices=["ensemble", "ema", "none"], default="none")
    ap.add_argument("--ensemble-decay", type=float, default=0.5)
    ap.add_argument("--ema-alpha", type=float, default=0.6)
    ap.add_argument("--cbf", action="store_true")
    ap.add_argument("--cbf-mode", choices=["project", "cone"], default="cone")
    ap.add_argument("--cbf-d-safe", type=float, default=0.75)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.6)
    ap.add_argument("--cbf-proj-iters", type=int, default=15)
    ap.add_argument("--cbf-proj-lr", type=float, default=0.08)
    ap.add_argument("--cbf-cone-margin", type=float, default=0.05)
    ap.add_argument("--cbf-trust", type=float, default=0.3)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    navdp_root = resolve_navdp_root(args.navdp_root)
    add_navdp_to_path(navdp_root)

    from navdp.data.habitat_route_dataset import _empty_belief_tensor, _proprio_from_pose
    from navdp.extensions import DepthObstacleMap, nearest_obstacle_point, project_chunk_cone, project_forward_velocity_cbf
    from rollout_habitat_policy import ActionSmoother, action_to_control, frame_to_spatial, load_model, resolve_modes, resolve_obstacle_channel

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    terrain = TerrainHeight(
        mode=args.terrain_height_mode,
        heightmap=Path(args.heightmap).expanduser().resolve() if args.heightmap else None,
        obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
        flat_y=args.flat_y,
        size_x=args.size_x,
        size_z=args.size_z,
        size_y=args.size_y,
        flip_x=args.flip_heightmap_x,
        flip_z=args.flip_heightmap_z,
        swap_xz=args.swap_heightmap_xz,
    )

    device = args.device
    model, train_args = load_model(Path(args.ckpt).expanduser().resolve(), device, args.weights)
    modes = resolve_modes(args, train_args)
    if modes["action_mode"] == "waypoint":
        raise ValueError("Mars rollout executes velocity actions; use action3d or action2d checkpoint/mode.")
    use_obstacle_channel = resolve_obstacle_channel(args, train_args)
    image_size = int(args.image_size or train_args.get("image_size", 224))
    intr = intrinsics_from_hfov(args.height, args.width, args.hfov_deg)
    obstacle_builder = DepthObstacleMap(camera_intrinsics=intr)
    smoother = ActionSmoother(args.action_smoothing, args.ensemble_decay, args.ema_alpha)

    sim = make_sim(Path(args.scene), args.height, args.width, args.hfov_deg)
    agent = sim.initialize_agent(0)

    x = float(args.start_x)
    z = float(args.start_z)
    yaw = math.radians(float(args.start_yaw_deg))
    dt = 1.0 / float(args.hz)
    goal_y = args.goal_y
    if goal_y is None:
        goal_y = terrain(float(args.goal_x), float(args.goal_z)) + float(args.goal_height)
    goal = np.asarray([float(args.goal_x), float(goal_y), float(args.goal_z)], dtype=np.float32)

    rows = {k: [] for k in [
        "rgb", "depth", "goal_mask", "obstacle_mask", "seg_masks", "pose", "proprio",
        "action_3d", "pred_chunk", "goal_visible_pixels", "goal_u", "goal_v", "goal_distance",
    ]}
    video_frames = []
    prev_obstacle_point = None
    cbf_active = 0

    print("Mars NavDP rollout", flush=True)
    print(f"  navdp_root : {navdp_root}", flush=True)
    print(f"  scene      : {Path(args.scene).expanduser().resolve()}", flush=True)
    print(f"  ckpt       : {Path(args.ckpt).expanduser().resolve()}", flush=True)
    print(f"  terrain    : {terrain.mode}", flush=True)
    print(f"  goal       : x={goal[0]:.2f} y={goal[1]:.2f} z={goal[2]:.2f}", flush=True)
    print(f"  modes      : action={modes['action_mode']} proprio={modes['proprio_mode']} obstacle_channel={use_obstacle_channel}", flush=True)

    try:
        for step in range(int(args.max_steps)):
            y = terrain(x, z) + float(args.clearance)
            position = np.asarray([x, y, z], dtype=np.float32)
            set_agent_pose(agent, x, y, z, yaw)
            obs = sim.get_sensor_observations()
            rgb, depth = rgb_depth(obs)
            goal_mask, goal_info = project_goal_mask(
                goal=goal,
                position=position,
                yaw=yaw,
                height=rgb.shape[0],
                width=rgb.shape[1],
                hfov_deg=args.hfov_deg,
                radius=args.goal_radius,
                clamp_to_edge=not args.no_clamp_goal_to_edge,
            )
            if args.obstacle_mode == "depth":
                obstacle_mask = depth_obstacle_mask(depth, args.obstacle_depth_threshold, args.obstacle_min_y_frac)
            else:
                obstacle_mask = np.zeros_like(goal_mask, dtype=np.uint8)

            spatial = frame_to_spatial(depth, goal_mask, image_size, obstacle_mask, include_obstacle_channel=use_obstacle_channel).to(device)
            obstacle_map = obstacle_builder.build(depth) if args.obstacle_mode == "depth" else np.zeros((96, 96), dtype=np.float32)
            obstacle_t = torch.from_numpy(obstacle_map[None]).float().to(device)

            qx, qy, qz, qw = yaw_quat_xyzw(yaw)
            pose = np.asarray([x, y, z, qx, qy, qz, qw], dtype=np.float32)
            proprio = _proprio_from_pose(pose, modes["proprio_mode"], planar_axes=(0, 2), yaw_axis=modes["yaw_axis"])
            proprio_t = torch.from_numpy(proprio[None]).float().to(device)
            belief_t = torch.from_numpy(_empty_belief_tensor()[None]).float().to(device)
            route_index = torch.zeros(1, dtype=torch.long, device=device)
            active_goal_index = torch.zeros(1, dtype=torch.long, device=device)

            pred = model.sample(
                spatial,
                proprio_t,
                steps=int(args.sample_steps),
                belief_tensor=belief_t,
                obstacle_map=obstacle_t,
                route_index=route_index,
                active_goal_index=active_goal_index,
            )

            obstacle_point = None
            if args.cbf and int(obstacle_mask.sum()) > 0:
                obstacle_point = nearest_obstacle_point(obstacle_mask, depth, intr)
                if obstacle_point is not None:
                    cbf_active += 1
                    v_o = np.zeros(2, dtype=np.float32)
                    if args.cbf_mode == "cone":
                        if args.zero_lateral and pred.shape[-1] >= 3:
                            pred = pred.clone()
                            pred[..., 1] = 0.0
                        p_lat = float(obstacle_point[1])
                        side = -1.0 if p_lat > 0.0 else 1.0
                        pred = project_chunk_cone(
                            pred,
                            obstacle_point,
                            v_o,
                            r=args.cbf_d_safe,
                            dt=dt,
                            vel_scale=1.0,
                            iters=args.cbf_proj_iters,
                            lr=args.cbf_proj_lr,
                            trust=args.cbf_trust,
                            margin=args.cbf_cone_margin,
                            deadzone_range=args.cbf_d_safe + args.cbf_deadzone,
                            side=side,
                        )
                    prev_obstacle_point = obstacle_point
            _ = prev_obstacle_point

            pred_chunk = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
            chunk_ctrl = np.stack([
                action_to_control(
                    a,
                    action_mode=modes["action_mode"],
                    max_forward_speed=args.max_forward_speed,
                    max_lateral_speed=args.max_lateral_speed,
                    max_yaw_rate=args.max_yaw_rate,
                )
                for a in pred_chunk
            ]).astype(np.float32)
            smoother.add(step, chunk_ctrl)
            action_3d = smoother.get(step)
            if args.zero_lateral and action_3d.shape[0] >= 2:
                action_3d = action_3d.copy()
                action_3d[1] = 0.0
            if args.cbf and args.cbf_mode == "project" and obstacle_point is not None:
                action_3d, _ = project_forward_velocity_cbf(
                    action_3d,
                    obstacle_point,
                    np.zeros(2, dtype=np.float32),
                    d_safe=args.cbf_d_safe,
                    gamma=args.cbf_gamma,
                    deadzone=args.cbf_deadzone,
                    trust=args.cbf_trust,
                )

            next_position, next_yaw = integrate_mars(position, yaw, action_3d, dt)
            x = float(np.clip(next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5))
            z = float(np.clip(next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5))
            yaw = wrap_angle(next_yaw)

            goal_dist = float(np.linalg.norm(goal[[0, 2]] - np.asarray([x, z], dtype=np.float32)))
            seg = np.zeros_like(goal_mask, dtype=np.uint8)
            seg[goal_mask > 0] = 1
            seg[obstacle_mask > 0] = 2

            rows["rgb"].append(rgb)
            rows["depth"].append(depth)
            rows["goal_mask"].append(goal_mask.astype(np.uint8))
            rows["obstacle_mask"].append(obstacle_mask.astype(np.uint8))
            rows["seg_masks"].append(seg.astype(np.uint8))
            rows["pose"].append(pose)
            rows["proprio"].append(proprio.astype(np.float32))
            rows["action_3d"].append(action_3d.astype(np.float32))
            rows["pred_chunk"].append(pred_chunk.astype(np.float32))
            rows["goal_visible_pixels"].append(int(goal_mask.sum()))
            rows["goal_u"].append(float(goal_info["u"]))
            rows["goal_v"].append(float(goal_info["v"]))
            rows["goal_distance"].append(goal_dist)

            if step % max(int(args.save_every), 1) == 0:
                text = f"t={step} dist={goal_dist:.2f} v={action_3d[0]:.2f} yaw={math.degrees(yaw):.1f}"
                frame = overlay_frame(rgb, goal_mask, obstacle_mask, text)
                frame.save(frame_dir / f"frame_{step:04d}.png")
                video_frames.append(frame)

            if step % 10 == 0:
                print(
                    f"step {step:04d} | dist={goal_dist:.2f} | goal_px={int(goal_mask.sum())} "
                    f"| action=[{action_3d[0]:.2f},{action_3d[1]:.2f},{action_3d[2]:.2f}]",
                    flush=True,
                )
            if goal_dist <= float(args.stop_dist):
                print(f"Reached goal at step {step} dist={goal_dist:.2f}m", flush=True)
                break
    finally:
        sim.close()

    success = bool(rows["goal_distance"] and rows["goal_distance"][-1] <= float(args.stop_dist))
    npz_path = out_dir / "rollout.npz"
    np.savez_compressed(
        npz_path,
        rgb=np.stack(rows["rgb"]).astype(np.uint8),
        depth=np.stack(rows["depth"]).astype(np.float32),
        goal_mask=np.stack(rows["goal_mask"]).astype(np.uint8),
        obstacle_mask=np.stack(rows["obstacle_mask"]).astype(np.uint8),
        seg_masks=np.stack(rows["seg_masks"]).astype(np.uint8),
        pose=np.stack(rows["pose"]).astype(np.float32),
        proprio=np.stack(rows["proprio"]).astype(np.float32),
        action_3d=np.stack(rows["action_3d"]).astype(np.float32),
        pred_chunk=np.stack(rows["pred_chunk"]).astype(np.float32),
        goal_visible_pixels=np.asarray(rows["goal_visible_pixels"], dtype=np.int32),
        goal_u=np.asarray(rows["goal_u"], dtype=np.float32),
        goal_v=np.asarray(rows["goal_v"], dtype=np.float32),
        goal_distance=np.asarray(rows["goal_distance"], dtype=np.float32),
        goal_position=goal.astype(np.float32),
        success=np.asarray(success, dtype=bool),
        hz=np.asarray(float(args.hz), dtype=np.float32),
    )
    manifest = {
        "success": success,
        "frames": len(rows["rgb"]),
        "final_distance": float(rows["goal_distance"][-1]) if rows["goal_distance"] else None,
        "goal_position": goal.tolist(),
        "ckpt": str(Path(args.ckpt).expanduser().resolve()),
        "scene": str(Path(args.scene).expanduser().resolve()),
        "terrain_mode": terrain.mode,
        "cbf_active": cbf_active,
        "npz": str(npz_path),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if args.save_video:
        save_video(video_frames, out_dir / "rollout.mp4", fps=max(float(args.hz) / max(int(args.save_every), 1), 1.0))
    print(f"Saved rollout: {npz_path}", flush=True)
    print(f"Output dir   : {out_dir}", flush=True)


def integrate_mars(position: np.ndarray, yaw: float, action_3d: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
    v_fwd, v_lat, yaw_rate = [float(x) for x in np.asarray(action_3d, dtype=np.float32).reshape(-1)[:3]]
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    out = np.asarray(position, dtype=np.float32).copy()
    out[0] += (fwd_x * v_fwd + left_x * v_lat) * float(dt)
    out[2] += (fwd_z * v_fwd + left_z * v_lat) * float(dt)
    return out, float(yaw + yaw_rate * float(dt))


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    main()

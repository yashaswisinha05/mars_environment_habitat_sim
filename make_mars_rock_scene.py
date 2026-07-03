"""make_mars_rock_scene.py - add visible goal/obstacle rocks to Mars terrain.

Creates a Habitat-friendly Y-up OBJ scene from marsyard2022.obj and appends two
procedural irregular rock meshes:

  - goal rock: bright green/tan, used with rollout --goal-x/--goal-z
  - obstacle rock: dark red/brown, used with rollout --ghost-obstacle-x/--ghost-obstacle-z

The rocks are real rendered geometry in RGB/depth. Masks remain synthetic in
rollout_navdp_policy.py, projected from the same world coordinates.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_TERRAIN_OBJ = HERE / "marsyard2022.obj"
DEFAULT_OUT_OBJ = HERE / "marsyard2022_rocks.obj"


class ObjWriter:
    def __init__(self, start_index: int):
        self.next_index = int(start_index) + 1
        self.lines: list[str] = []

    def object(self, name: str) -> None:
        self.lines.append(f"\no {name}\n")

    def usemtl(self, name: str) -> None:
        self.lines.append(f"usemtl {name}\n")

    def add_vertices(self, vertices: Sequence[Sequence[float]]) -> list[int]:
        ids = []
        for v in vertices:
            ids.append(self.next_index)
            self.lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
            self.next_index += 1
        return ids

    def face(self, ids: Sequence[int]) -> None:
        self.lines.append("f " + " ".join(str(int(i)) for i in ids) + "\n")


def parse_terrain_grid(path: Path) -> tuple[list[float], list[float], list[list[float]], int]:
    verts = []
    vertex_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(parts[1])
            z_axis = float(parts[2])
            height = float(parts[3])
            verts.append((x, z_axis, height))
            vertex_count += 1
    if not verts:
        raise RuntimeError(f"no terrain vertices found in {path}")

    xs = sorted({v[0] for v in verts})
    zs = sorted({v[1] for v in verts})
    x_to_i = {v: i for i, v in enumerate(xs)}
    z_to_i = {v: i for i, v in enumerate(zs)}
    grid = [[None for _ in xs] for _ in zs]
    heights = []
    for x, z, h in verts:
        grid[z_to_i[z]][x_to_i[x]] = h
        heights.append(h)
    fill = sum(heights) / max(len(heights), 1)
    clean = []
    for row in grid:
        clean.append([fill if v is None else float(v) for v in row])
    return xs, zs, clean, vertex_count


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def terrain_height(xs: list[float], zs: list[float], grid: list[list[float]], x: float, z: float) -> float:
    xx = _clip(x, xs[0], xs[-1])
    zz = _clip(z, zs[0], zs[-1])
    col = max(0, min(len(xs) - 2, bisect.bisect_right(xs, xx) - 1))
    row = max(0, min(len(zs) - 2, bisect.bisect_right(zs, zz) - 1))
    x0, x1 = float(xs[col]), float(xs[col + 1])
    z0, z1 = float(zs[row]), float(zs[row + 1])
    tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
    tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
    h00 = float(grid[row][col])
    h10 = float(grid[row][col + 1])
    h01 = float(grid[row + 1][col])
    h11 = float(grid[row + 1][col + 1])
    h0 = h00 * (1.0 - tx) + h10 * tx
    h1 = h01 * (1.0 - tx) + h11 * tx
    return float(h0 * (1.0 - tz) + h1 * tz)


def local_height_max(xs: list[float], zs: list[float], grid: list[list[float]], x: float, z: float, radius: float) -> float:
    radius = max(float(radius), 0.0)
    vals = [terrain_height(xs, zs, grid, x, z)]
    if radius <= 1e-6:
        return vals[0]
    for dx in (-radius, -0.5 * radius, 0.0, 0.5 * radius, radius):
        for dz in (-radius, -0.5 * radius, 0.0, 0.5 * radius, radius):
            if dx * dx + dz * dz <= radius * radius + 1e-8:
                vals.append(terrain_height(xs, zs, grid, x + dx, z + dz))
    return max(vals)


def write_materials(src_mtl: Path, out_mtl: Path) -> None:
    base = src_mtl.read_text(encoding="utf-8", errors="ignore") if src_mtl.exists() else ""
    extra = """

newmtl goal_rock_green
Ka 0.030 0.280 0.060
Kd 0.080 0.680 0.160
Ks 0.030 0.030 0.030
d 1.0
illum 2

newmtl obstacle_rock_red
Ka 0.320 0.060 0.030
Kd 0.620 0.130 0.070
Ks 0.030 0.020 0.020
d 1.0
illum 2
"""
    out_mtl.write_text(base.rstrip() + extra, encoding="ascii")
    if src_mtl.exists():
        for line in base.splitlines():
            if not line.strip().lower().startswith("map_kd"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            src = src_mtl.parent / parts[1].strip()
            dst = out_mtl.parent / parts[1].strip()
            if src.exists() and src.resolve() != dst.resolve() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except PermissionError:
                    pass


def add_rock(
    writer: ObjWriter,
    *,
    name: str,
    material: str,
    x: float,
    ground_y: float,
    z: float,
    radius: float,
    height: float,
    seed: int,
    rings: int = 4,
    segments: int = 14,
) -> None:
    """Append an irregular low-poly ellipsoid rock sitting on ground_y."""
    rng = random.Random(int(seed))
    writer.object(name)
    writer.usemtl(material)

    verts: list[tuple[float, float, float]] = []
    # bottom center, rings from low to high, top cap
    verts.append((float(x), float(ground_y), float(z)))
    for r in range(1, int(rings) + 1):
        v = r / float(int(rings) + 1)
        theta = v * math.pi
        ring_y = ground_y + height * (0.08 + 0.90 * v)
        ring_radius = radius * math.sin(theta)
        for s in range(int(segments)):
            a = 2.0 * math.pi * s / int(segments)
            jitter = 0.78 + 0.42 * rng.random()
            y_jitter = (rng.random() - 0.5) * 0.10 * height
            sx = 1.0 + 0.20 * math.sin(1.7 * a + seed)
            sz = 0.85 + 0.22 * math.cos(1.3 * a + 0.7 * seed)
            vx = x + ring_radius * sx * jitter * math.cos(a)
            vy = ring_y + y_jitter
            vz = z + ring_radius * sz * jitter * math.sin(a)
            verts.append((vx, vy, vz))
    verts.append((float(x), float(ground_y + height), float(z)))
    ids = writer.add_vertices(verts)

    n = int(segments)
    bottom = ids[0]
    top = ids[-1]
    first_ring = 1
    last_ring = 1 + (int(rings) - 1) * n

    # bottom fan
    for s in range(n):
        writer.face([bottom, ids[first_ring + s], ids[first_ring + ((s + 1) % n)]])

    # sides
    for r in range(int(rings) - 1):
        a0 = 1 + r * n
        a1 = 1 + (r + 1) * n
        for s in range(n):
            writer.face([
                ids[a0 + s],
                ids[a0 + ((s + 1) % n)],
                ids[a1 + ((s + 1) % n)],
                ids[a1 + s],
            ])

    # top fan
    for s in range(n):
        writer.face([ids[last_ring + s], top, ids[last_ring + ((s + 1) % n)]])


def write_scene(
    *,
    terrain_obj: Path,
    out_obj: Path,
    goal_x: float,
    goal_z: float,
    obstacle_x: float,
    obstacle_z: float,
    goal_radius: float,
    goal_height: float,
    obstacle_radius: float,
    obstacle_height: float,
    terrain_radius: float,
    seed: int,
) -> dict:
    xs, zs, grid, vertex_count = parse_terrain_grid(terrain_obj)
    goal_ground = local_height_max(xs, zs, grid, goal_x, goal_z, terrain_radius)
    obs_ground = local_height_max(xs, zs, grid, obstacle_x, obstacle_z, terrain_radius)

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    out_mtl = out_obj.with_suffix(".mtl")
    write_materials(terrain_obj.with_suffix(".mtl"), out_mtl)

    writer = ObjWriter(vertex_count)
    with terrain_obj.open("r", encoding="utf-8", errors="ignore") as fin, out_obj.open("w", encoding="ascii") as fout:
        fout.write(f"mtllib {out_mtl.name}\n")
        for line in fin:
            if line.startswith("mtllib "):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    # hm2obj.py writes x / row-axis / height. Rewrite to x / y-up-height / z.
                    tx = float(parts[1])
                    tz = float(parts[2])
                    ty = float(parts[3])
                    fout.write(f"v {tx:.6f} {ty:.6f} {tz:.6f}\n")
                    continue
            fout.write(line)

        add_rock(
            writer,
            name="goal_rock",
            material="goal_rock_green",
            x=goal_x,
            ground_y=goal_ground,
            z=goal_z,
            radius=goal_radius,
            height=goal_height,
            seed=seed + 11,
        )
        add_rock(
            writer,
            name="obstacle_rock",
            material="obstacle_rock_red",
            x=obstacle_x,
            ground_y=obs_ground,
            z=obstacle_z,
            radius=obstacle_radius,
            height=obstacle_height,
            seed=seed + 97,
        )
        fout.writelines(writer.lines)

    manifest = {
        "out_obj": str(out_obj),
        "out_mtl": str(out_mtl),
        "terrain_obj": str(terrain_obj),
        "goal_rock": {"x": goal_x, "y": goal_ground, "z": goal_z, "radius": goal_radius, "height": goal_height},
        "obstacle_rock": {"x": obstacle_x, "y": obs_ground, "z": obstacle_z, "radius": obstacle_radius, "height": obstacle_height},
    }
    out_json = out_obj.with_suffix(".json")
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(out_json)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a Mars scene with real procedural rocks for goal and obstacle.")
    ap.add_argument("--terrain-obj", default=str(DEFAULT_TERRAIN_OBJ))
    ap.add_argument("--out", default=str(DEFAULT_OUT_OBJ))
    ap.add_argument("--goal-x", type=float, default=8.0)
    ap.add_argument("--goal-z", type=float, default=-8.0)
    ap.add_argument("--obstacle-x", type=float, default=4.0)
    ap.add_argument("--obstacle-z", type=float, default=0.0)
    ap.add_argument("--goal-radius", type=float, default=0.75)
    ap.add_argument("--goal-height", type=float, default=1.0)
    ap.add_argument("--obstacle-radius", type=float, default=0.95)
    ap.add_argument("--obstacle-height", type=float, default=1.2)
    ap.add_argument("--terrain-radius", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    result = write_scene(
        terrain_obj=Path(args.terrain_obj).expanduser().resolve(),
        out_obj=Path(args.out).expanduser().resolve(),
        goal_x=float(args.goal_x),
        goal_z=float(args.goal_z),
        obstacle_x=float(args.obstacle_x),
        obstacle_z=float(args.obstacle_z),
        goal_radius=float(args.goal_radius),
        goal_height=float(args.goal_height),
        obstacle_radius=float(args.obstacle_radius),
        obstacle_height=float(args.obstacle_height),
        terrain_radius=float(args.terrain_radius),
        seed=int(args.seed),
    )
    print("Wrote Mars rock scene:")
    print("  OBJ:", result["out_obj"])
    print("  MTL:", result["out_mtl"])
    print("  manifest:", result["manifest"])
    print("  goal rock:", result["goal_rock"])
    print("  obstacle rock:", result["obstacle_rock"])
    print("\nRollout example:")
    print(
        "python rollout_navdp_policy.py "
        f"--scene {Path(result['out_obj']).name} --terrain-obj marsyard2022.obj "
        f"--goal-x {args.goal_x:g} --goal-z {args.goal_z:g} "
        f"--ghost-obstacle-x {args.obstacle_x:g} --ghost-obstacle-z {args.obstacle_z:g} "
        f"--ghost-obstacle-world-radius {args.obstacle_radius:g} "
        "--habitat-use-obstacle-channel --sample-steps 30 --action-smoothing ensemble "
        "--scene-height-flip-z --clearance 1.4 --pose-terrain-radius 0.8 "
        "--goal-height 1.2 --goal-terrain-radius 0.8 --lost-goal-ghost "
        "--ghost-obstacle-bypass --ghost-obstacle-bypass-clearance 1.8 "
        "--cbf --cbf-active-range 6.0 --cbf-mode cone --zero-lateral --cbf-metric mahalanobis "
        "--cbf-cov-mode shrink --cbf-radius-mode perceived --robot-radius 0.25 "
        "--safety-margin 0.15 --cbf-proj-iters 40 --cbf-keep-speed 1.0"
    )


if __name__ == "__main__":
    main()

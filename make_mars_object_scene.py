"""make_mars_object_scene.py - add simple object meshes to the Mars terrain.

The original terrain OBJ is written by hm2obj.py as x / row-axis / height. This
script rewrites it into a Habitat-friendly Y-up OBJ (x / height / z) and appends
a simple procedural chair mesh by default. A cup obstacle can be added only when
requested with --with-cup.

The resulting OBJ can be passed to rollout_navdp_policy.py via --scene while the
chair world coordinate is passed as --goal-x/--goal-z for the ghost goal mask.
"""
from __future__ import annotations

import argparse
import bisect
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence, Tuple



HERE = Path(__file__).resolve().parent
DEFAULT_TERRAIN_OBJ = HERE / "marsyard2022.obj"
DEFAULT_OUT_OBJ = HERE / "marsyard2022_chair.obj"


class ObjWriter:
    def __init__(self, start_index: int):
        self.next_index = int(start_index) + 1
        self.lines: list[str] = []

    def usemtl(self, name: str) -> None:
        self.lines.append(f"usemtl {name}\n")

    def object(self, name: str) -> None:
        self.lines.append(f"\no {name}\n")

    def add_vertices(self, vertices: Sequence[Sequence[float]]) -> list[int]:
        ids = []
        for v in vertices:
            ids.append(self.next_index)
            self.lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
            self.next_index += 1
        return ids

    def face(self, ids: Sequence[int]) -> None:
        self.lines.append("f " + " ".join(str(int(i)) for i in ids) + "\n")

    def extend(self, lines: Iterable[str]) -> None:
        self.lines.extend(lines)


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
    for x, z, height in verts:
        grid[z_to_i[z]][x_to_i[x]] = height
        heights.append(height)
    fill = sum(heights) / max(len(heights), 1)
    clean_grid: list[list[float]] = []
    for row in grid:
        clean_grid.append([fill if v is None else float(v) for v in row])
    return xs, zs, clean_grid, vertex_count


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


def rotate_xz(x: float, z: float, yaw: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x + s * z, -s * x + c * z


def box_vertices(center: Sequence[float], size: Sequence[float], yaw: float = 0.0) -> list[tuple[float, float, float]]:
    cx, cy, cz = [float(v) for v in center]
    sx, sy, sz = [float(v) for v in size]
    out = []
    for dx in (-sx / 2.0, sx / 2.0):
        for dy in (-sy / 2.0, sy / 2.0):
            for dz in (-sz / 2.0, sz / 2.0):
                rx, rz = rotate_xz(dx, dz, yaw)
                out.append((cx + rx, cy + dy, cz + rz))
    return out


def add_box(writer: ObjWriter, name: str, material: str, center: Sequence[float], size: Sequence[float], yaw: float = 0.0) -> None:
    writer.object(name)
    writer.usemtl(material)
    ids = writer.add_vertices(box_vertices(center, size, yaw))
    # Vertex order follows nested dx,dy,dz loops.
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    for f in faces:
        writer.face([ids[i] for i in f])


def add_chair(writer: ObjWriter, x: float, ground_y: float, z: float, yaw: float, scale: float) -> None:
    s = float(scale)
    seat_y = ground_y + 0.47 * s
    add_box(writer, "chair_seat", "chair_green", (x, seat_y, z), (0.95 * s, 0.16 * s, 0.85 * s), yaw)
    back_local_z = 0.34 * s
    bx, bz = rotate_xz(0.0, back_local_z, yaw)
    add_box(writer, "chair_back", "chair_green", (x + bx, ground_y + 0.90 * s, z + bz), (0.95 * s, 0.85 * s, 0.14 * s), yaw)
    for lx in (-0.36 * s, 0.36 * s):
        for lz in (-0.30 * s, 0.30 * s):
            px, pz = rotate_xz(lx, lz, yaw)
            add_box(writer, "chair_leg", "chair_dark", (x + px, ground_y + 0.23 * s, z + pz), (0.12 * s, 0.46 * s, 0.12 * s), yaw)


def add_cylinder(
    writer: ObjWriter,
    name: str,
    material: str,
    x: float,
    ground_y: float,
    z: float,
    radius: float,
    height: float,
    segments: int,
) -> None:
    writer.object(name)
    writer.usemtl(material)
    bottom_y = float(ground_y)
    top_y = float(ground_y + height)
    verts = []
    for y in (bottom_y, top_y):
        for i in range(int(segments)):
            a = 2.0 * math.pi * i / int(segments)
            verts.append((x + radius * math.cos(a), y, z + radius * math.sin(a)))
    ids = writer.add_vertices(verts)
    n = int(segments)
    for i in range(n):
        j = (i + 1) % n
        writer.face([ids[i], ids[j], ids[n + j], ids[n + i]])
    bottom_center = writer.add_vertices([(x, bottom_y, z)])[0]
    for i in range(n):
        j = (i + 1) % n
        writer.face([bottom_center, ids[j], ids[i]])


def add_cup(writer: ObjWriter, x: float, ground_y: float, z: float, yaw: float, scale: float) -> None:
    s = float(scale)
    radius = 0.26 * s
    height = 0.70 * s
    add_cylinder(writer, "cup_body", "cup_red", x, ground_y, z, radius, height, 32)
    # A simple side handle made from three cuboids, large enough to be visible.
    hx, hz = rotate_xz(radius + 0.08 * s, 0.0, yaw)
    add_box(writer, "cup_handle_vert", "cup_red", (x + hx, ground_y + 0.38 * s, z + hz), (0.10 * s, 0.42 * s, 0.08 * s), yaw)
    for hy in (0.20 * s, 0.56 * s):
        add_box(writer, "cup_handle_link", "cup_red", (x + hx * 0.75, ground_y + hy, z + hz * 0.75), (0.24 * s, 0.08 * s, 0.08 * s), yaw)


def write_materials(src_mtl: Path, out_mtl: Path) -> None:
    base = src_mtl.read_text(encoding="utf-8", errors="ignore") if src_mtl.exists() else ""
    extra = """

newmtl chair_green
Ka 0.000 0.350 0.060
Kd 0.000 0.650 0.120
Ks 0.050 0.050 0.050
d 1.0
illum 2

newmtl chair_dark
Ka 0.020 0.020 0.020
Kd 0.080 0.080 0.070
Ks 0.020 0.020 0.020
d 1.0
illum 2

newmtl cup_red
Ka 0.500 0.020 0.020
Kd 0.900 0.050 0.040
Ks 0.100 0.100 0.100
d 1.0
illum 2
"""
    out_mtl.write_text(base.rstrip() + extra, encoding="ascii")
    if src_mtl.exists():
        tex_names = []
        for line in base.splitlines():
            if line.strip().lower().startswith("map_kd"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    tex_names.append(parts[1].strip())
        for tex in tex_names:
            src = src_mtl.parent / tex
            dst = out_mtl.parent / tex
            if src.exists() and src.resolve() != dst.resolve():
                try:
                    if not dst.exists():
                        shutil.copy2(src, dst)
                except PermissionError:
                    # Another smoke/run may already be copying the same large texture.
                    # The OBJ/MTL remain valid as long as the file exists by load time.
                    pass


def write_scene(
    *,
    terrain_obj: Path,
    out_obj: Path,
    chair_x: float,
    chair_z: float,
    cup_x: float,
    cup_z: float,
    chair_yaw: float,
    cup_yaw: float,
    chair_scale: float,
    cup_scale: float,
    include_cup: bool,
) -> dict:
    xs, zs, grid, vertex_count = parse_terrain_grid(terrain_obj)
    chair_ground = terrain_height(xs, zs, grid, chair_x, chair_z)
    cup_ground = terrain_height(xs, zs, grid, cup_x, cup_z) if include_cup else None
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    out_mtl = out_obj.with_suffix(".mtl")
    src_mtl = terrain_obj.with_suffix(".mtl")
    write_materials(src_mtl, out_mtl)

    writer = ObjWriter(vertex_count)
    with terrain_obj.open("r", encoding="utf-8", errors="ignore") as fin, out_obj.open("w", encoding="ascii") as fout:
        fout.write(f"mtllib {out_mtl.name}\n")
        for line in fin:
            if line.startswith("mtllib "):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x = float(parts[1])
                    z_axis = float(parts[2])
                    height = float(parts[3])
                    fout.write(f"v {x:.6f} {height:.6f} {z_axis:.6f}\n")
                    continue
            fout.write(line)

        add_chair(writer, chair_x, chair_ground, chair_z, math.radians(chair_yaw), chair_scale)
        if include_cup and cup_ground is not None:
            add_cup(writer, cup_x, cup_ground, cup_z, math.radians(cup_yaw), cup_scale)
        fout.writelines(writer.lines)

    return {
        "out_obj": str(out_obj),
        "out_mtl": str(out_mtl),
        "chair": {"x": chair_x, "y": chair_ground, "z": chair_z},
        "cup": ({"x": cup_x, "y": cup_ground, "z": cup_z} if include_cup and cup_ground is not None else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a Mars terrain scene with a chair goal and optional cup obstacle.")
    ap.add_argument("--terrain-obj", default=str(DEFAULT_TERRAIN_OBJ))
    ap.add_argument("--out", default=str(DEFAULT_OUT_OBJ))
    ap.add_argument("--chair-x", type=float, default=8.0)
    ap.add_argument("--chair-z", type=float, default=-8.0)
    ap.add_argument("--cup-x", type=float, default=5.0)
    ap.add_argument("--cup-z", type=float, default=2.0)
    ap.add_argument("--chair-yaw-deg", type=float, default=0.0)
    ap.add_argument("--cup-yaw-deg", type=float, default=0.0)
    ap.add_argument("--chair-scale", type=float, default=1.25)
    ap.add_argument("--cup-scale", type=float, default=1.8)
    ap.add_argument("--with-cup", action="store_true", help="Add the cup obstacle mesh. Default is chair-only / no obstacle.")
    args = ap.parse_args()

    result = write_scene(
        terrain_obj=Path(args.terrain_obj).expanduser().resolve(),
        out_obj=Path(args.out).expanduser().resolve(),
        chair_x=float(args.chair_x),
        chair_z=float(args.chair_z),
        cup_x=float(args.cup_x),
        cup_z=float(args.cup_z),
        chair_yaw=float(args.chair_yaw_deg),
        cup_yaw=float(args.cup_yaw_deg),
        chair_scale=float(args.chair_scale),
        cup_scale=float(args.cup_scale),
        include_cup=bool(args.with_cup),
    )
    print("Wrote Mars object scene:")
    print("  OBJ:", result["out_obj"])
    print("  MTL:", result["out_mtl"])
    print("  chair:", result["chair"])
    if result["cup"] is not None:
        print("  cup:", result["cup"])
    print("\nRollout example:")
    cmd = (
        "python rollout_navdp_policy.py --scene "
        f"{Path(result['out_obj']).name} --terrain-obj marsyard2022.obj "
        f"--goal-x {args.chair_x:g} --goal-z {args.chair_z:g} "
        "--habitat-use-obstacle-channel --sample-steps 30 --action-smoothing ensemble "
        "--cbf --cbf-mode cone --zero-lateral --cbf-metric mahalanobis "
        "--cbf-cov-mode shrink --cbf-radius-mode perceived --robot-radius 0.25 "
        "--safety-margin 0.15 --cbf-proj-iters 40 --cbf-keep-speed 1.0 "
        "--lost-goal-ghost --replan-every 3"
    )
    if result["cup"] is not None:
        cmd += f" --ghost-obstacle-x {args.cup_x:g} --ghost-obstacle-z {args.cup_z:g}"
    print(cmd)


if __name__ == "__main__":
    main()

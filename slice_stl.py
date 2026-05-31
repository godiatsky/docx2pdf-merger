#!/usr/bin/env python3
"""
Slice an STL file into horizontal contour layers (topographic map / sculpture effect).

Usage:
    python slice_stl.py input.stl output.stl
    python slice_stl.py input.stl output.stl --slices 60 --gap 0.5
    python slice_stl.py input.stl output.stl --axis Y --slices 80

Options:
    --slices N      Number of slices (default: 80)
    --gap R         Gap ratio 0-1, fraction of pitch that is empty (default: 0.45)
    --axis A        Slice axis: X, Y, or Z (default: Z)
"""

import argparse
import sys
import numpy as np

try:
    import trimesh
except ImportError:
    print("Error: trimesh not installed. Run: pip install trimesh numpy shapely")
    sys.exit(1)


def slice_mesh(mesh, axis: str, num_slices: int, gap_ratio: float) -> list:
    """Return a list of thin slab meshes along the given axis."""
    ax = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
    normal = np.zeros(3)
    normal[ax] = 1.0

    lo, hi = mesh.bounds[0][ax], mesh.bounds[1][ax]
    pitch = (hi - lo) / num_slices
    thickness = pitch * (1.0 - gap_ratio)

    slabs = []
    for i in range(num_slices):
        center = lo + (i + 0.5) * pitch
        origin = np.zeros(3)
        origin[ax] = center

        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
            if section is None:
                continue

            path2d, transform = section.to_2D()
            if path2d is None or len(path2d.entities) == 0:
                continue

            slab = path2d.extrude(thickness)

            # Center the slab around Z=0, then rotate back to world space
            slab.apply_translation([0, 0, -thickness / 2])
            slab.apply_transform(np.linalg.inv(transform))

            slabs.append(slab)

            if (i + 1) % 20 == 0 or i + 1 == num_slices:
                print(f"  {i + 1}/{num_slices} slices done", flush=True)

        except Exception:
            continue

    return slabs


def main():
    parser = argparse.ArgumentParser(
        description="Slice STL into horizontal contour layers"
    )
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output STL file")
    parser.add_argument(
        "--slices", type=int, default=80, metavar="N",
        help="Number of slices (default: 80)"
    )
    parser.add_argument(
        "--gap", type=float, default=0.45, metavar="R",
        help="Gap ratio 0-1 — fraction of pitch that is empty (default: 0.45)"
    )
    parser.add_argument(
        "--axis", default="Z", choices=["X", "Y", "Z"],
        help="Slice axis (default: Z)"
    )
    args = parser.parse_args()

    if not (0.0 < args.gap < 1.0):
        parser.error("--gap must be between 0 and 1 (exclusive)")

    print(f"Loading {args.input} ...")
    mesh = trimesh.load(args.input, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        print("Error: could not load a valid mesh from the file.")
        sys.exit(1)

    print(
        f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n"
        f"Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}\n"
        f"Slicing along {args.axis.upper()}-axis into {args.slices} layers "
        f"(gap ratio {args.gap}) ..."
    )

    slabs = slice_mesh(mesh, args.axis, args.slices, args.gap)

    if not slabs:
        print("Error: no slices were generated. Check that the mesh is valid.")
        sys.exit(1)

    print(f"Combining {len(slabs)} slabs ...")
    combined = trimesh.util.concatenate(slabs)
    combined.export(args.output)
    print(f"Saved → {args.output}")
    print(
        f"Result: {len(combined.vertices)} vertices, {len(combined.faces)} faces"
    )


if __name__ == "__main__":
    main()

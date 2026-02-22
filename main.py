#!/usr/bin/env python3
"""
main.py — CLI entry point for The Shape Index.

Four modes:
    --analyze    Standard shape analysis of a single surface
    --compare    Compare all six preset surfaces
    --complementarity  Lock-and-Key complementarity test
    --saddle     Saddle-point detection & catalogue

Usage examples:
    python main.py --analyze --surface sphere
    python main.py --compare --save
    python main.py --complementarity --save
    python main.py --saddle --surface torus --save
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.shape_engine import (
    get_preset_surfaces,
    build_double_sphere,
    build_binding_pocket,
    build_bump,
)
from src.analysis import (
    analyze_shape,
    analyze_patches,
    analyze_complementarity,
    analyze_saddle_points,
    compare_preset_surfaces,
    shape_summary,
)
from src.visualization import MatplotlibRenderer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════

SURFACE_MAP = {
    "sphere": "Sphere",
    "ellipsoid": "Ellipsoid",
    "saddle": "Saddle",
    "torus": "Torus",
    "wavy": "Wavy Surface",
    "pocket": "Binding Pocket",
}

SURFACE_CHOICES = list(SURFACE_MAP.keys())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shape_index",
        description="The Shape Index — Differential Geometry of Binding",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", default=True,
                      help="Analyse a single surface (default).")
    mode.add_argument("--compare", action="store_true",
                      help="Compare all six preset surfaces.")
    mode.add_argument("--complementarity", action="store_true",
                      help="Run Lock-and-Key complementarity test.")
    mode.add_argument("--saddle", action="store_true",
                      help="Saddle-point detection and catalogue.")

    p.add_argument("--surface", choices=SURFACE_CHOICES, default="ellipsoid",
                   help="Preset surface (default: ellipsoid).")
    p.add_argument("--save", action="store_true",
                   help="Save figures to figures/.")
    p.add_argument("--verbose", action="store_true",
                   help="Print verbose summary.")
    return p


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_mesh(name: str):
    presets = get_preset_surfaces()
    key = SURFACE_MAP.get(name, name)
    if key in presets:
        return presets[key]
    raise ValueError(f"Unknown surface: {name}")


def _ensure_figures():
    os.makedirs("figures", exist_ok=True)


def _save_fig(fig, name: str):
    _ensure_figures()
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════


def cmd_analyze(args):
    print("\n═══ Shape Analysis ═══")
    mesh = _get_mesh(args.surface)
    print(f"Surface: {mesh.name}")
    print(f"Vertices: {mesh.n_vertices:,}  |  Faces: {mesh.n_faces:,}")

    t0 = time.perf_counter()
    analysis = analyze_shape(mesh)
    dt = time.perf_counter() - t0
    print(f"Analysis completed in {dt:.3f} s\n")

    if args.verbose:
        print(shape_summary(analysis))

    # Figures
    fig1 = MatplotlibRenderer.shape_index_surface(
        analysis.shape_result,
        title=f"Shape Index — {mesh.name}",
    )
    fig2 = MatplotlibRenderer.category_bar(analysis)
    fig3 = MatplotlibRenderer.si_histogram(analysis.shape_result)
    fig4 = MatplotlibRenderer.curvature_scatter(analysis.shape_result)

    if args.save:
        _save_fig(fig1, f"shape_index_{args.surface}.png")
        _save_fig(fig2, f"category_bar_{args.surface}.png")
        _save_fig(fig3, f"si_histogram_{args.surface}.png")
        _save_fig(fig4, f"curvature_scatter_{args.surface}.png")
    else:
        plt.show()


def cmd_compare(args):
    print("\n═══ Preset Surface Comparison ═══")
    t0 = time.perf_counter()
    comparison = compare_preset_surfaces()
    dt = time.perf_counter() - t0
    print(f"Compared {len(comparison.analyses)} surfaces in {dt:.3f} s\n")

    # Print table
    header = f"{'Surface':20s} {'Verts':>6s} {'Faces':>6s} {'Area':>8s} {'Mean SI':>8s} {'Mean C':>8s} {'Dominant':>15s} {'Patches':>8s} {'Saddles':>8s}"
    print(header)
    print("─" * len(header))
    for row in comparison.summary_table:
        print(
            f"{row['Surface']:20s} {row['Vertices']:6d} {row['Faces']:6d} "
            f"{row['Area']:>8s} {row['Mean SI']:>8s} {row['Mean C']:>8s} "
            f"{row['Dominant']:>15s} {row['Patches']:>8d} {row['Saddles']:>8d}"
        )
    print()

    fig = MatplotlibRenderer.preset_comparison(comparison)
    if args.save:
        _save_fig(fig, "preset_comparison.png")
    else:
        plt.show()


def cmd_complementarity(args):
    print("\n═══ Lock-and-Key Complementarity Test ═══")
    protein = build_binding_pocket(
        pocket_depth=1.2, pocket_radius=1.8, surface_radius=2.2,
        name="Protein Pocket",
    )
    ligand = build_bump(
        bump_height=1.2, bump_radius=1.8, surface_radius=2.2,
        name="Ligand Bump",
    )
    print(f"Protein: {protein.name}  ({protein.n_vertices} vertices)")
    print(f"Ligand:  {ligand.name}  ({ligand.n_vertices} vertices)")

    t0 = time.perf_counter()
    comp = analyze_complementarity(protein, ligand)
    dt = time.perf_counter() - t0
    print(f"Analysis completed in {dt:.3f} s\n")
    print(comp.explanation)

    fig = MatplotlibRenderer.complementarity_histograms(comp)
    if args.save:
        _save_fig(fig, "complementarity_histograms.png")
    else:
        plt.show()


def cmd_saddle(args):
    print("\n═══ Saddle-Point Detection ═══")
    mesh = _get_mesh(args.surface)
    print(f"Surface: {mesh.name}")

    t0 = time.perf_counter()
    analysis = analyze_shape(mesh)
    catalogue = analyze_saddle_points(analysis)
    dt = time.perf_counter() - t0
    print(f"Detection completed in {dt:.3f} s\n")
    print(catalogue.explanation)

    # Show saddle overlay with shape index surface
    fig = MatplotlibRenderer.shape_index_surface(
        analysis.shape_result,
        title=f"Saddle Points — {mesh.name}",
    )
    if args.save:
        _save_fig(fig, f"saddle_{args.surface}.png")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.complementarity:
        cmd_complementarity(args)
    elif args.compare:
        cmd_compare(args)
    elif args.saddle:
        cmd_saddle(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()

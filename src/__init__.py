"""
The Shape Index — Differential Geometry of Binding
====================================================

Week 14, Project 2 — Biophysics Portfolio
Ryan Kamp · University of Cincinnati

Submodules
----------
shape_engine
    Core differential geometry engine: mesh generation, principal
    curvature estimation, Koenderink Shape Index, Curvedness,
    patch segmentation, saddle-point detection, complementarity
    analysis for protein–ligand binding.
analysis
    Higher-level analysis pipelines: full shape analysis, patch
    statistics, complementarity scoring, saddle-point catalogue,
    preset surface comparison.
visualization
    Dual rendering: PlotlyRenderer (interactive 3-D shape-indexed
    surfaces, patch maps, complementarity views) and
    MatplotlibRenderer (static publication-quality figures).
"""

from __future__ import annotations

# ── Core engine ──────────────────────────────────────────────────────
from src.shape_engine import (
    Vertex, TriMesh, CurvatureResult, ShapeIndexResult,
    PatchResult, SaddlePoint, ComplementarityResult,
    DEFAULT_PATCH_BINS, SHAPE_CATEGORIES,
    build_ellipsoid, build_saddle, build_torus, build_sphere,
    build_double_sphere, build_wavy_surface,
    estimate_vertex_normals, estimate_principal_curvatures,
    compute_shape_index, compute_curvedness,
    classify_shape, segment_patches,
    detect_saddle_points, complementarity_analysis,
    get_preset_surfaces,
)

# ── Analysis pipelines ──────────────────────────────────────────────
from src.analysis import (
    FullShapeAnalysis, PatchStatistics, ComplementarityAnalysis,
    SaddlePointCatalogue, PresetComparisonResult,
    analyze_shape, analyze_patches, analyze_complementarity,
    analyze_saddle_points, compare_preset_surfaces, shape_summary,
)

# ── Visualization ────────────────────────────────────────────────────
from src.visualization import (PlotlyRenderer, MatplotlibRenderer)

__all__ = [
    # Engine types
    "Vertex", "TriMesh", "CurvatureResult", "ShapeIndexResult",
    "PatchResult", "SaddlePoint", "ComplementarityResult",
    "DEFAULT_PATCH_BINS", "SHAPE_CATEGORIES",
    # Mesh factories
    "build_ellipsoid", "build_saddle", "build_torus", "build_sphere",
    "build_double_sphere", "build_wavy_surface",
    # Core algorithms
    "estimate_vertex_normals", "estimate_principal_curvatures",
    "compute_shape_index", "compute_curvedness",
    "classify_shape", "segment_patches",
    "detect_saddle_points", "complementarity_analysis",
    "get_preset_surfaces",
    # Analysis
    "FullShapeAnalysis", "PatchStatistics", "ComplementarityAnalysis",
    "SaddlePointCatalogue", "PresetComparisonResult",
    "analyze_shape", "analyze_patches", "analyze_complementarity",
    "analyze_saddle_points", "compare_preset_surfaces", "shape_summary",
    # Visualization
    "PlotlyRenderer", "MatplotlibRenderer",
]

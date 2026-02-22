"""
analysis.py — Higher-level Analysis Pipelines
================================================

Builds on shape_engine to provide structured analysis objects:
FullShapeAnalysis, PatchStatistics, ComplementarityAnalysis,
SaddlePointCatalogue, and PresetComparisonResult.

Each pipeline function returns a rich dataclass that the CLI,
Streamlit app, and test suite consume.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.shape_engine import (
    TriMesh,
    CurvatureResult,
    ShapeIndexResult,
    PatchResult,
    SaddlePoint,
    ComplementarityResult,
    SHAPE_CATEGORIES,
    estimate_principal_curvatures,
    compute_shape_index,
    compute_curvedness,
    classify_shape,
    classify_shape_array,
    compute_full_shape_index,
    segment_patches,
    detect_saddle_points,
    complementarity_analysis,
    get_preset_surfaces,
    build_double_sphere,
    build_binding_pocket,
    build_bump,
    DEFAULT_PATCH_BINS,
)


# ═══════════════════════════════════════════════════════════════════════
# Analysis result dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FullShapeAnalysis:
    """Complete shape analysis of a single surface.

    Fields
    ------
    mesh : TriMesh
    shape_result : ShapeIndexResult
    patches : PatchResult
    saddle_points : list[SaddlePoint]
    category_fractions : dict[str, float]
    mean_shape_index : float
    std_shape_index : float
    mean_curvedness : float
    std_curvedness : float
    mean_gaussian : float
    mean_mean_curv : float
    total_area : float
    explanation : str
    """
    mesh: TriMesh
    shape_result: ShapeIndexResult
    patches: PatchResult
    saddle_points: List[SaddlePoint]
    category_fractions: Dict[str, float]
    mean_shape_index: float
    std_shape_index: float
    mean_curvedness: float
    std_curvedness: float
    mean_gaussian: float
    mean_mean_curv: float
    total_area: float
    explanation: str


@dataclass
class PatchStatistics:
    """Statistics about the patch segmentation.

    Fields
    ------
    n_patches : int
    largest_patch_id : int
    largest_patch_area : float
    largest_patch_category : str
    category_patch_counts : dict[str, int]
    mean_patch_area : float
    std_patch_area : float
    explanation : str
    """
    n_patches: int
    largest_patch_id: int
    largest_patch_area: float
    largest_patch_category: str
    category_patch_counts: Dict[str, int]
    mean_patch_area: float
    std_patch_area: float
    explanation: str


@dataclass
class ComplementarityAnalysis:
    """Enriched complementarity result with statistics.

    Fields
    ------
    comp_result : ComplementarityResult
    protein_analysis : FullShapeAnalysis
    ligand_analysis : FullShapeAnalysis
    protein_mean_si : float
    ligand_mean_si : float
    mirror_score : float
        How well the SI histograms mirror each other around zero.
    explanation : str
    """
    comp_result: ComplementarityResult
    protein_analysis: FullShapeAnalysis
    ligand_analysis: FullShapeAnalysis
    protein_mean_si: float
    ligand_mean_si: float
    mirror_score: float
    explanation: str


@dataclass
class SaddlePointCatalogue:
    """Catalogue of detected saddle points with context.

    Fields
    ------
    saddle_points : list[SaddlePoint]
    n_saddle : int
    fraction_saddle : float
    mean_curvedness_at_saddle : float
    spatial_clusters : list[NDArray]
    explanation : str
    """
    saddle_points: List[SaddlePoint]
    n_saddle: int
    fraction_saddle: float
    mean_curvedness_at_saddle: float
    spatial_clusters: List[NDArray]
    explanation: str


@dataclass
class PresetComparisonResult:
    """Side-by-side comparison of all preset surfaces.

    Fields
    ------
    analyses : dict[str, FullShapeAnalysis]
    summary_table : list[dict]
    explanation : str
    """
    analyses: Dict[str, FullShapeAnalysis]
    summary_table: List[dict]
    explanation: str


# ═══════════════════════════════════════════════════════════════════════
# Pipeline functions
# ═══════════════════════════════════════════════════════════════════════


def analyze_shape(mesh: TriMesh, n_bins: int = DEFAULT_PATCH_BINS) -> FullShapeAnalysis:
    """Run full shape analysis on a single mesh.

    Returns a FullShapeAnalysis with curvature, Shape Index,
    patch segmentation, saddle points, and summary statistics.
    """
    sr = compute_full_shape_index(mesh)
    patches = segment_patches(mesh, sr.shape_index, n_bins=n_bins)
    saddles = detect_saddle_points(sr, patches=patches)

    # Category fractions
    cat_counts: Dict[str, int] = {name: 0 for name in SHAPE_CATEGORIES}
    for cat in sr.categories:
        if cat in cat_counts:
            cat_counts[cat] += 1
    total = max(len(sr.categories), 1)
    cat_fracs = {k: v / total for k, v in cat_counts.items()}

    mean_si = float(np.mean(sr.shape_index))
    std_si = float(np.std(sr.shape_index))
    mean_C = float(np.mean(sr.curvedness))
    std_C = float(np.std(sr.curvedness))
    mean_K = float(np.mean(sr.curvature.gaussian_curvature))
    mean_H = float(np.mean(sr.curvature.mean_curvature))
    area = mesh.total_area()

    # Build explanation
    dominant = max(cat_fracs, key=cat_fracs.get)  # type: ignore
    explanation = (
        f"Surface: {mesh.name}\n"
        f"Vertices: {mesh.n_vertices:,}  |  Faces: {mesh.n_faces:,}  |  "
        f"Area: {area:.2f}\n\n"
        f"Shape Index  — mean: {mean_si:+.3f}, std: {std_si:.3f}\n"
        f"Curvedness   — mean: {mean_C:.3f}, std: {std_C:.3f}\n"
        f"Gaussian K   — mean: {mean_K:.4f}\n"
        f"Mean H       — mean: {mean_H:.4f}\n\n"
        f"Dominant category: {dominant} ({cat_fracs[dominant]:.1%})\n"
        f"Patches: {patches.n_patches}  |  "
        f"Saddle points: {len(saddles)}\n"
    )

    return FullShapeAnalysis(
        mesh=mesh,
        shape_result=sr,
        patches=patches,
        saddle_points=saddles,
        category_fractions=cat_fracs,
        mean_shape_index=mean_si,
        std_shape_index=std_si,
        mean_curvedness=mean_C,
        std_curvedness=std_C,
        mean_gaussian=mean_K,
        mean_mean_curv=mean_H,
        total_area=area,
        explanation=explanation,
    )


def analyze_patches(analysis: FullShapeAnalysis) -> PatchStatistics:
    """Compute patch-level statistics from a FullShapeAnalysis."""
    patches = analysis.patches
    areas = list(patches.patch_areas.values())
    if not areas:
        return PatchStatistics(
            n_patches=0, largest_patch_id=-1, largest_patch_area=0.0,
            largest_patch_category="N/A",
            category_patch_counts={}, mean_patch_area=0.0,
            std_patch_area=0.0, explanation="No patches found.",
        )

    largest_id = max(patches.patch_areas, key=patches.patch_areas.get)  # type: ignore
    largest_area = patches.patch_areas[largest_id]
    largest_cat = patches.patch_categories.get(largest_id, "N/A")

    cat_counts: Dict[str, int] = {name: 0 for name in SHAPE_CATEGORIES}
    for pid, cat in patches.patch_categories.items():
        if cat in cat_counts:
            cat_counts[cat] += 1

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))

    explanation = (
        f"Total patches: {patches.n_patches}\n"
        f"Largest patch: #{largest_id} — {largest_cat} "
        f"(area = {largest_area:.2f})\n"
        f"Mean patch area: {mean_area:.3f} ± {std_area:.3f}\n\n"
        f"Patch counts by category:\n"
    )
    for cat, cnt in cat_counts.items():
        if cnt > 0:
            explanation += f"  {cat}: {cnt}\n"

    return PatchStatistics(
        n_patches=patches.n_patches,
        largest_patch_id=largest_id,
        largest_patch_area=largest_area,
        largest_patch_category=largest_cat,
        category_patch_counts=cat_counts,
        mean_patch_area=mean_area,
        std_patch_area=std_area,
        explanation=explanation,
    )


def analyze_complementarity(
    protein_mesh: TriMesh,
    ligand_mesh: TriMesh,
) -> ComplementarityAnalysis:
    """Run the Lock-and-Key complementarity test.

    Returns an enriched ComplementarityAnalysis.
    """
    comp = complementarity_analysis(protein_mesh, ligand_mesh)

    prot_an = analyze_shape(protein_mesh)
    lig_an = analyze_shape(ligand_mesh)

    prot_mean = float(np.mean(comp.protein_result.shape_index))
    lig_mean = float(np.mean(comp.ligand_result.shape_index))

    # Mirror score: correlation of flipped histograms
    p_hist = comp.protein_si_histogram
    l_hist = comp.ligand_si_histogram
    l_flipped = l_hist[::-1]
    if np.std(p_hist) > 0 and np.std(l_flipped) > 0:
        mirror = float(np.corrcoef(p_hist, l_flipped)[0, 1])
    else:
        mirror = 0.0
    mirror = max(0.0, mirror)

    explanation = comp.explanation + (
        f"\n\nProtein mean SI: {prot_mean:+.3f} (should be negative / concave)\n"
        f"Ligand  mean SI: {lig_mean:+.3f} (should be positive / convex)\n"
        f"Histogram mirror score: {mirror:.3f}\n"
    )

    return ComplementarityAnalysis(
        comp_result=comp,
        protein_analysis=prot_an,
        ligand_analysis=lig_an,
        protein_mean_si=prot_mean,
        ligand_mean_si=lig_mean,
        mirror_score=mirror,
        explanation=explanation,
    )


def analyze_saddle_points(
    analysis: FullShapeAnalysis,
    cluster_radius: float = 1.0,
) -> SaddlePointCatalogue:
    """Build a catalogue of saddle points with spatial clustering.

    Saddle points within *cluster_radius* of each other are grouped.
    """
    saddles = analysis.saddle_points
    n = len(saddles)
    frac = n / max(analysis.mesh.n_vertices, 1)

    if n == 0:
        return SaddlePointCatalogue(
            saddle_points=[], n_saddle=0, fraction_saddle=0.0,
            mean_curvedness_at_saddle=0.0, spatial_clusters=[],
            explanation="No saddle points detected.",
        )

    mean_C = float(np.mean([s.curvedness for s in saddles]))

    # Simple greedy clustering
    positions = np.array([s.position for s in saddles])
    visited = set()
    clusters: List[NDArray] = []
    for i in range(n):
        if i in visited:
            continue
        dists = np.linalg.norm(positions - positions[i], axis=1)
        cluster_mask = dists < cluster_radius
        cluster_indices = np.where(cluster_mask)[0]
        for ci in cluster_indices:
            visited.add(int(ci))
        centroid = positions[cluster_mask].mean(axis=0)
        clusters.append(centroid)

    explanation = (
        f"Saddle points: {n} ({frac:.1%} of vertices)\n"
        f"Mean curvedness at saddle: {mean_C:.4f}\n"
        f"Spatial clusters (r = {cluster_radius}): {len(clusters)}\n\n"
        "In protein surfaces, saddle regions often correspond to:\n"
        "  • Hinge regions between domains\n"
        "  • Transition-state stabilisation sites\n"
        "  • Channel entrances\n"
    )

    return SaddlePointCatalogue(
        saddle_points=saddles,
        n_saddle=n,
        fraction_saddle=frac,
        mean_curvedness_at_saddle=mean_C,
        spatial_clusters=clusters,
        explanation=explanation,
    )


def compare_preset_surfaces() -> PresetComparisonResult:
    """Analyse all six preset surfaces and build a comparison table."""
    presets = get_preset_surfaces()
    analyses: Dict[str, FullShapeAnalysis] = {}
    rows: List[dict] = []

    for name, mesh in presets.items():
        an = analyze_shape(mesh)
        analyses[name] = an
        dominant = max(an.category_fractions, key=an.category_fractions.get)  # type: ignore
        rows.append({
            "Surface": name,
            "Vertices": mesh.n_vertices,
            "Faces": mesh.n_faces,
            "Area": f"{an.total_area:.1f}",
            "Mean SI": f"{an.mean_shape_index:+.3f}",
            "Std SI": f"{an.std_shape_index:.3f}",
            "Mean C": f"{an.mean_curvedness:.3f}",
            "Dominant": dominant,
            "Patches": an.patches.n_patches,
            "Saddles": len(an.saddle_points),
        })

    explanation = (
        "Comparison of six preset surfaces:\n\n"
        "• Sphere     — uniform positive curvature (all Cap)\n"
        "• Ellipsoid  — varying curvature from Cap to Saddle Ridge\n"
        "• Saddle     — hyperbolic paraboloid (pure saddle)\n"
        "• Torus      — mixed: outer ring = Ridge/Dome, inner = Rut/Trough\n"
        "• Wavy       — alternating Cups and Caps\n"
        "• Pocket     — concave centre (Cup) with flat surround\n"
    )

    return PresetComparisonResult(
        analyses=analyses,
        summary_table=rows,
        explanation=explanation,
    )


def shape_summary(analysis: FullShapeAnalysis) -> str:
    """Return a human-readable text summary of a shape analysis."""
    lines = [
        f"═══ Shape Analysis: {analysis.mesh.name} ═══",
        f"",
        f"Mesh:  {analysis.mesh.n_vertices:,} vertices, "
        f"{analysis.mesh.n_faces:,} faces, "
        f"area = {analysis.total_area:.2f}",
        f"",
        f"Shape Index (S):",
        f"  Mean: {analysis.mean_shape_index:+.4f}",
        f"  Std : {analysis.std_shape_index:.4f}",
        f"",
        f"Curvedness (C):",
        f"  Mean: {analysis.mean_curvedness:.4f}",
        f"  Std : {analysis.std_curvedness:.4f}",
        f"",
        f"Gaussian curvature (K):  mean = {analysis.mean_gaussian:.6f}",
        f"Mean curvature (H):      mean = {analysis.mean_mean_curv:.6f}",
        f"",
        f"Category fractions:",
    ]
    for cat, frac in analysis.category_fractions.items():
        bar = "█" * int(frac * 30)
        lines.append(f"  {cat:15s} {frac:6.1%}  {bar}")
    lines.append(f"")
    lines.append(f"Patches: {analysis.patches.n_patches}")
    lines.append(f"Saddle points: {len(analysis.saddle_points)}")
    lines.append(f"")
    return "\n".join(lines)

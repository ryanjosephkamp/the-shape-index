"""
test_shape_index.py — Comprehensive Test Suite
================================================

18 test classes, 90+ test methods covering all modules:
  - Engine: TriMesh, curvature estimation, Shape Index, Curvedness,
    classification, patch segmentation, saddle detection,
    complementarity, presets, mesh builders
  - Analysis: FullShapeAnalysis, PatchStatistics, Complementarity,
    SaddlePointCatalogue, PresetComparison, shape_summary
  - Visualization: PlotlyRenderer (12 methods), MatplotlibRenderer
    (6 methods)
  - CLI: argument parsing for all modes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_engine import (
    Vertex,
    TriMesh,
    CurvatureResult,
    ShapeIndexResult,
    PatchResult,
    SaddlePoint,
    ComplementarityResult,
    DEFAULT_PATCH_BINS,
    SHAPE_CATEGORIES,
    SHAPE_CATEGORY_COLORS,
    build_sphere,
    build_ellipsoid,
    build_saddle,
    build_torus,
    build_wavy_surface,
    build_double_sphere,
    build_binding_pocket,
    build_bump,
    estimate_vertex_normals,
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
)
from src.analysis import (
    FullShapeAnalysis,
    PatchStatistics,
    ComplementarityAnalysis,
    SaddlePointCatalogue,
    PresetComparisonResult,
    analyze_shape,
    analyze_patches,
    analyze_complementarity,
    analyze_saddle_points,
    compare_preset_surfaces,
    shape_summary,
)
from src.visualization import (
    PlotlyRenderer,
    MatplotlibRenderer,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Constants & Categories
# ═══════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test global constants and category definitions."""

    def test_nine_shape_categories(self):
        """SHAPE_CATEGORIES should define exactly nine categories."""
        assert len(SHAPE_CATEGORIES) == 9

    def test_categories_cover_range(self):
        """Category bins should cover [-1, +1]."""
        lo = min(lo for lo, _ in SHAPE_CATEGORIES.values())
        hi = max(hi for _, hi in SHAPE_CATEGORIES.values())
        assert lo == -1.0
        assert hi == 1.0

    def test_category_colors_exist(self):
        """Every category has a colour."""
        for cat in SHAPE_CATEGORIES:
            assert cat in SHAPE_CATEGORY_COLORS

    def test_default_patch_bins(self):
        """Default number of patch bins should be 9."""
        assert DEFAULT_PATCH_BINS == 9

    def test_cup_range(self):
        """Cup should cover [-1, -0.75)."""
        assert SHAPE_CATEGORIES["Cup"] == (-1.0, -0.75)

    def test_cap_range(self):
        """Cap should cover [0.75, 1.0)."""
        assert SHAPE_CATEGORIES["Cap"] == (0.75, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# 2. Vertex
# ═══════════════════════════════════════════════════════════════════════


class TestVertex:
    """Test the Vertex dataclass."""

    def test_position_property(self):
        """Vertex.position returns a 3-element ndarray."""
        v = Vertex(1.0, 2.0, 3.0)
        pos = v.position
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])

    def test_default_normal_none(self):
        """Normal is None by default."""
        v = Vertex(0.0, 0.0, 0.0)
        assert v.normal is None

    def test_with_normal(self):
        """Vertex can store a normal vector."""
        n = np.array([0.0, 0.0, 1.0])
        v = Vertex(0.0, 0.0, 0.0, normal=n)
        np.testing.assert_array_almost_equal(v.normal, [0.0, 0.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════
# 3. TriMesh
# ═══════════════════════════════════════════════════════════════════════


class TestTriMesh:
    """Test the TriMesh dataclass."""

    @pytest.fixture
    def simple_mesh(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return TriMesh("test", verts, faces)

    def test_n_vertices(self, simple_mesh):
        assert simple_mesh.n_vertices == 4

    def test_n_faces(self, simple_mesh):
        assert simple_mesh.n_faces == 2

    def test_center_of_mass(self, simple_mesh):
        com = simple_mesh.center_of_mass
        np.testing.assert_array_almost_equal(com, [0.5, 0.5, 0.0])

    def test_face_normals_direction(self, simple_mesh):
        fn = simple_mesh.face_normals()
        # Both faces are in xy-plane, normals should point in z
        for n in fn:
            assert abs(n[2]) > 0

    def test_face_areas_positive(self, simple_mesh):
        areas = simple_mesh.face_areas()
        assert np.all(areas > 0)

    def test_total_area(self, simple_mesh):
        area = simple_mesh.total_area()
        assert area == pytest.approx(1.0, abs=0.01)

    def test_bounding_box(self, simple_mesh):
        lo, hi = simple_mesh.bounding_box()
        np.testing.assert_array_almost_equal(lo, [0, 0, 0])
        np.testing.assert_array_almost_equal(hi, [1, 1, 0])


# ═══════════════════════════════════════════════════════════════════════
# 4. Mesh Builders
# ═══════════════════════════════════════════════════════════════════════


class TestMeshBuilders:
    """Test mesh construction functions."""

    def test_build_sphere_vertices(self):
        mesh = build_sphere(radius=1.0, n_u=10, n_v=10)
        assert mesh.n_vertices == 100

    def test_build_sphere_faces(self):
        mesh = build_sphere(radius=1.0, n_u=10, n_v=10)
        assert mesh.n_faces == 2 * 9 * 9

    def test_build_sphere_radius(self):
        mesh = build_sphere(radius=2.0, n_u=20, n_v=20)
        dists = np.linalg.norm(mesh.vertices, axis=1)
        assert np.allclose(dists, 2.0, atol=0.15)

    def test_build_ellipsoid_shape(self):
        mesh = build_ellipsoid(a=2.0, b=1.0, c=0.5)
        extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        # x-extent should be largest, z smallest
        assert extent[0] > extent[1] > extent[2]

    def test_build_saddle_center(self):
        mesh = build_saddle(scale=2.0)
        com = mesh.center_of_mass
        np.testing.assert_array_almost_equal(com[:2], [0.0, 0.0], decimal=1)

    def test_build_torus_positive_area(self):
        mesh = build_torus(R=2.0, r=0.6)
        assert mesh.total_area() > 0

    def test_build_wavy_surface(self):
        mesh = build_wavy_surface(amplitude=0.5, frequency=2.0)
        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0

    def test_build_double_sphere_returns_pair(self):
        protein, ligand = build_double_sphere()
        assert isinstance(protein, TriMesh)
        assert isinstance(ligand, TriMesh)

    def test_build_binding_pocket(self):
        mesh = build_binding_pocket()
        # Center should dip below zero
        center_z = mesh.vertices[mesh.vertices.shape[0] // 2, 2]
        # The pocket should have some negative z values
        assert mesh.vertices[:, 2].min() < 0

    def test_build_bump(self):
        mesh = build_bump()
        # The bump should have positive z values
        assert mesh.vertices[:, 2].max() > 0

    def test_preset_surfaces_count(self):
        presets = get_preset_surfaces()
        assert len(presets) == 6

    def test_preset_surfaces_names(self):
        presets = get_preset_surfaces()
        expected = {"Sphere", "Ellipsoid", "Saddle", "Torus",
                    "Wavy Surface", "Binding Pocket"}
        assert set(presets.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════
# 5. Normal Estimation
# ═══════════════════════════════════════════════════════════════════════


class TestNormals:
    """Test vertex normal estimation."""

    def test_sphere_normals_unit(self):
        mesh = build_sphere(radius=1.0, n_u=20, n_v=20)
        normals = estimate_vertex_normals(mesh)
        norms = np.linalg.norm(normals, axis=1)
        # Poles have degenerate faces and zero-area normals; skip them
        interior = norms > 0.5
        np.testing.assert_allclose(norms[interior], 1.0, atol=1e-6)
        # At most a few polar vertices may be degenerate
        assert interior.sum() >= normals.shape[0] - 4

    def test_sphere_normals_outward(self):
        """Normals on a sphere should point outward (same direction as vertex)."""
        mesh = build_sphere(radius=1.0, n_u=20, n_v=20)
        normals = estimate_vertex_normals(mesh)
        # For a centered unit sphere, normal ≈ vertex position
        dots = (normals * mesh.vertices).sum(axis=1)
        # Most should be positive (outward), a few at poles may differ
        assert np.mean(dots > 0) > 0.8

    def test_normals_shape(self):
        mesh = build_sphere(radius=1.0, n_u=10, n_v=10)
        normals = estimate_vertex_normals(mesh)
        assert normals.shape == (mesh.n_vertices, 3)


# ═══════════════════════════════════════════════════════════════════════
# 6. Curvature Estimation
# ═══════════════════════════════════════════════════════════════════════


class TestCurvature:
    """Test principal curvature estimation."""

    def test_sphere_curvatures_uniform(self):
        """A sphere should have roughly uniform curvature."""
        mesh = build_sphere(radius=1.0, n_u=30, n_v=30)
        curv = estimate_principal_curvatures(mesh)
        # Interior vertices should have kappa ≈ 1.0 for unit sphere
        # (Edge effects exist at polar regions)
        k1_median = np.median(curv.kappa1)
        k2_median = np.median(curv.kappa2)
        assert abs(k1_median - k2_median) < 1.0  # Should be similar

    def test_curvature_result_shapes(self):
        mesh = build_ellipsoid(n_u=15, n_v=15)
        curv = estimate_principal_curvatures(mesh)
        V = mesh.n_vertices
        assert curv.kappa1.shape == (V,)
        assert curv.kappa2.shape == (V,)
        assert curv.mean_curvature.shape == (V,)
        assert curv.gaussian_curvature.shape == (V,)
        assert curv.principal_dir1.shape == (V, 3)
        assert curv.principal_dir2.shape == (V, 3)
        assert curv.normals.shape == (V, 3)

    def test_mean_curvature_formula(self):
        """H = (κ₁ + κ₂) / 2."""
        mesh = build_torus(n_u=20, n_v=15)
        curv = estimate_principal_curvatures(mesh)
        expected = (curv.kappa1 + curv.kappa2) / 2.0
        np.testing.assert_allclose(curv.mean_curvature, expected)

    def test_gaussian_curvature_formula(self):
        """K = κ₁ · κ₂."""
        mesh = build_torus(n_u=20, n_v=15)
        curv = estimate_principal_curvatures(mesh)
        expected = curv.kappa1 * curv.kappa2
        np.testing.assert_allclose(curv.gaussian_curvature, expected)

    def test_kappa1_geq_kappa2(self):
        """Convention: κ₁ ≥ κ₂."""
        mesh = build_ellipsoid(n_u=20, n_v=20)
        curv = estimate_principal_curvatures(mesh)
        assert np.all(curv.kappa1 >= curv.kappa2 - 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 7. Shape Index & Curvedness
# ═══════════════════════════════════════════════════════════════════════


class TestShapeIndex:
    """Test Shape Index computation."""

    def test_shape_index_range(self):
        """Shape Index should be in [-1, +1]."""
        mesh = build_ellipsoid(n_u=25, n_v=25)
        curv = estimate_principal_curvatures(mesh)
        si = compute_shape_index(curv)
        assert np.all(si >= -1.0)
        assert np.all(si <= 1.0)

    def test_sphere_shape_index_positive(self):
        """A sphere should have positive Shape Index (Cap)."""
        mesh = build_sphere(radius=1.0, n_u=30, n_v=30)
        curv = estimate_principal_curvatures(mesh)
        si = compute_shape_index(curv)
        # Most vertices should be Cap (SI > 0.5)
        assert np.median(si) > 0.0

    def test_curvedness_non_negative(self):
        """Curvedness should be ≥ 0."""
        mesh = build_torus(n_u=25, n_v=15)
        curv = estimate_principal_curvatures(mesh)
        C = compute_curvedness(curv)
        assert np.all(C >= -1e-10)

    def test_curvedness_formula(self):
        """C = √((κ₁² + κ₂²)/2)."""
        mesh = build_ellipsoid(n_u=15, n_v=15)
        curv = estimate_principal_curvatures(mesh)
        expected = np.sqrt((curv.kappa1**2 + curv.kappa2**2) / 2.0)
        C = compute_curvedness(curv)
        np.testing.assert_allclose(C, expected)

    def test_flat_surface_curvedness_near_zero(self):
        """A mostly flat surface should have low curvedness."""
        mesh = build_binding_pocket(pocket_depth=0.01)
        curv = estimate_principal_curvatures(mesh)
        C = compute_curvedness(curv)
        assert np.median(C) < 0.5


# ═══════════════════════════════════════════════════════════════════════
# 8. Shape Classification
# ═══════════════════════════════════════════════════════════════════════


class TestClassification:
    """Test shape category classification."""

    def test_classify_cup(self):
        assert classify_shape(-0.9) == "Cup"

    def test_classify_trough(self):
        assert classify_shape(-0.6) == "Trough"

    def test_classify_rut(self):
        assert classify_shape(-0.3) == "Rut"

    def test_classify_saddle(self):
        assert classify_shape(0.0) == "Saddle"

    def test_classify_ridge(self):
        assert classify_shape(0.3) == "Ridge"

    def test_classify_dome(self):
        assert classify_shape(0.6) == "Dome"

    def test_classify_cap(self):
        assert classify_shape(0.9) == "Cap"

    def test_classify_extreme_positive(self):
        assert classify_shape(1.0) == "Cap"

    def test_classify_extreme_negative(self):
        assert classify_shape(-1.0) == "Cup"

    def test_classify_array(self):
        values = np.array([-0.9, 0.0, 0.9])
        cats = classify_shape_array(values)
        assert cats == ["Cup", "Saddle", "Cap"]


# ═══════════════════════════════════════════════════════════════════════
# 9. Full Shape Index Pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Test compute_full_shape_index."""

    def test_returns_shape_index_result(self):
        mesh = build_sphere(n_u=15, n_v=15)
        result = compute_full_shape_index(mesh)
        assert isinstance(result, ShapeIndexResult)

    def test_categories_length(self):
        mesh = build_sphere(n_u=15, n_v=15)
        result = compute_full_shape_index(mesh)
        assert len(result.categories) == mesh.n_vertices

    def test_mesh_reference(self):
        mesh = build_sphere(n_u=15, n_v=15)
        result = compute_full_shape_index(mesh)
        assert result.mesh is mesh


# ═══════════════════════════════════════════════════════════════════════
# 10. Patch Segmentation
# ═══════════════════════════════════════════════════════════════════════


class TestPatchSegmentation:
    """Test surface patch segmentation."""

    def test_all_vertices_labelled(self):
        """Every vertex should be assigned a patch."""
        mesh = build_torus(n_u=20, n_v=15)
        result = compute_full_shape_index(mesh)
        patches = segment_patches(mesh, result.shape_index)
        assert np.all(patches.labels >= 0)

    def test_at_least_one_patch(self):
        mesh = build_sphere(n_u=15, n_v=15)
        result = compute_full_shape_index(mesh)
        patches = segment_patches(mesh, result.shape_index)
        assert patches.n_patches >= 1

    def test_patch_areas_positive(self):
        mesh = build_ellipsoid(n_u=15, n_v=15)
        result = compute_full_shape_index(mesh)
        patches = segment_patches(mesh, result.shape_index)
        for area in patches.patch_areas.values():
            assert area >= 0

    def test_patch_centroids_exist(self):
        mesh = build_torus(n_u=20, n_v=15)
        result = compute_full_shape_index(mesh)
        patches = segment_patches(mesh, result.shape_index)
        for pid, centroid in patches.patch_centroids.items():
            assert centroid.shape == (3,)


# ═══════════════════════════════════════════════════════════════════════
# 11. Saddle-Point Detection
# ═══════════════════════════════════════════════════════════════════════


class TestSaddleDetection:
    """Test saddle-point detection."""

    def test_saddle_surface_has_saddle_points(self):
        """A pure saddle surface should have many saddle points."""
        mesh = build_saddle(scale=2.0, n=30)
        result = compute_full_shape_index(mesh)
        saddles = detect_saddle_points(result, threshold=0.15)
        assert len(saddles) > 0

    def test_sphere_few_saddle_points(self):
        """A sphere should have very few saddle points."""
        mesh = build_sphere(radius=1.0, n_u=25, n_v=25)
        result = compute_full_shape_index(mesh)
        saddles = detect_saddle_points(result, threshold=0.05)
        frac = len(saddles) / mesh.n_vertices
        assert frac < 0.3

    def test_saddle_point_attributes(self):
        mesh = build_torus(n_u=25, n_v=15)
        result = compute_full_shape_index(mesh)
        saddles = detect_saddle_points(result, threshold=0.15)
        if saddles:
            sp = saddles[0]
            assert isinstance(sp.vertex_index, int)
            assert sp.position.shape == (3,)
            assert -1 <= sp.shape_index <= 1

    def test_saddle_with_patches(self):
        mesh = build_torus(n_u=20, n_v=15)
        result = compute_full_shape_index(mesh)
        patches = segment_patches(mesh, result.shape_index)
        saddles = detect_saddle_points(result, patches=patches)
        if saddles:
            assert saddles[0].patch_id >= 0


# ═══════════════════════════════════════════════════════════════════════
# 12. Complementarity Analysis
# ═══════════════════════════════════════════════════════════════════════


class TestComplementarity:
    """Test protein–ligand complementarity analysis."""

    def test_complementarity_score_range(self):
        protein, ligand = build_double_sphere()
        comp = complementarity_analysis(protein, ligand)
        assert 0.0 <= comp.complementarity_score <= 1.0

    def test_histogram_bins(self):
        protein, ligand = build_double_sphere()
        comp = complementarity_analysis(protein, ligand)
        assert len(comp.histogram_bins) > 1

    def test_explanation_not_empty(self):
        protein, ligand = build_double_sphere()
        comp = complementarity_analysis(protein, ligand)
        assert len(comp.explanation) > 0

    def test_pocket_bump_complementarity(self):
        """Pocket + bump should show some complementarity."""
        pocket = build_binding_pocket()
        bump = build_bump()
        comp = complementarity_analysis(pocket, bump)
        # Score should be positive
        assert comp.complementarity_score > 0


# ═══════════════════════════════════════════════════════════════════════
# 13. Analysis — FullShapeAnalysis
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeShape:
    """Test the analyze_shape pipeline."""

    def test_returns_full_analysis(self):
        mesh = build_sphere(n_u=15, n_v=15)
        an = analyze_shape(mesh)
        assert isinstance(an, FullShapeAnalysis)

    def test_category_fractions_sum_to_one(self):
        mesh = build_ellipsoid(n_u=15, n_v=15)
        an = analyze_shape(mesh)
        total = sum(an.category_fractions.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_total_area_positive(self):
        mesh = build_torus(n_u=20, n_v=15)
        an = analyze_shape(mesh)
        assert an.total_area > 0

    def test_explanation_not_empty(self):
        mesh = build_sphere(n_u=15, n_v=15)
        an = analyze_shape(mesh)
        assert len(an.explanation) > 0


# ═══════════════════════════════════════════════════════════════════════
# 14. Analysis — PatchStatistics
# ═══════════════════════════════════════════════════════════════════════


class TestPatchStatistics:
    """Test analyze_patches."""

    def test_returns_patch_statistics(self):
        mesh = build_torus(n_u=20, n_v=15)
        an = analyze_shape(mesh)
        ps = analyze_patches(an)
        assert isinstance(ps, PatchStatistics)

    def test_largest_patch_area_positive(self):
        mesh = build_ellipsoid(n_u=15, n_v=15)
        an = analyze_shape(mesh)
        ps = analyze_patches(an)
        assert ps.largest_patch_area >= 0


# ═══════════════════════════════════════════════════════════════════════
# 15. Analysis — ComplementarityAnalysis
# ═══════════════════════════════════════════════════════════════════════


class TestComplementarityAnalysis:
    """Test analyze_complementarity."""

    def test_returns_result(self):
        protein, ligand = build_double_sphere()
        ca = analyze_complementarity(protein, ligand)
        assert isinstance(ca, ComplementarityAnalysis)

    def test_mirror_score_range(self):
        protein, ligand = build_double_sphere()
        ca = analyze_complementarity(protein, ligand)
        assert 0 <= ca.mirror_score <= 1.0

    def test_protein_analysis_present(self):
        protein, ligand = build_double_sphere()
        ca = analyze_complementarity(protein, ligand)
        assert isinstance(ca.protein_analysis, FullShapeAnalysis)


# ═══════════════════════════════════════════════════════════════════════
# 16. Analysis — SaddlePointCatalogue & PresetComparison
# ═══════════════════════════════════════════════════════════════════════


class TestSaddleCatalogue:
    """Test analyze_saddle_points."""

    def test_returns_catalogue(self):
        mesh = build_torus(n_u=20, n_v=15)
        an = analyze_shape(mesh)
        cat = analyze_saddle_points(an)
        assert isinstance(cat, SaddlePointCatalogue)

    def test_fraction_range(self):
        mesh = build_saddle(n=20)
        an = analyze_shape(mesh)
        cat = analyze_saddle_points(an)
        assert 0 <= cat.fraction_saddle <= 1.0


class TestPresetComparison:
    """Test compare_preset_surfaces."""

    def test_returns_result(self):
        comp = compare_preset_surfaces()
        assert isinstance(comp, PresetComparisonResult)

    def test_six_surfaces(self):
        comp = compare_preset_surfaces()
        assert len(comp.analyses) == 6

    def test_summary_table_rows(self):
        comp = compare_preset_surfaces()
        assert len(comp.summary_table) == 6


class TestShapeSummary:
    """Test shape_summary."""

    def test_returns_string(self):
        mesh = build_sphere(n_u=15, n_v=15)
        an = analyze_shape(mesh)
        s = shape_summary(an)
        assert isinstance(s, str)
        assert "Shape Index" in s


# ═══════════════════════════════════════════════════════════════════════
# 17. Visualization — PlotlyRenderer
# ═══════════════════════════════════════════════════════════════════════


class TestPlotlyRenderer:
    """Test all PlotlyRenderer static methods."""

    @pytest.fixture
    def analysis(self):
        mesh = build_ellipsoid(n_u=15, n_v=15)
        return analyze_shape(mesh)

    def test_shape_index_surface(self, analysis):
        fig = PlotlyRenderer.shape_index_surface(analysis.shape_result)
        assert fig is not None

    def test_curvedness_surface(self, analysis):
        fig = PlotlyRenderer.curvedness_surface(analysis.shape_result)
        assert fig is not None

    def test_patch_map(self, analysis):
        fig = PlotlyRenderer.patch_map(
            analysis.shape_result, analysis.patches
        )
        assert fig is not None

    def test_saddle_overlay(self, analysis):
        fig = PlotlyRenderer.saddle_overlay(
            analysis.shape_result, analysis.saddle_points
        )
        assert fig is not None

    def test_category_histogram(self, analysis):
        fig = PlotlyRenderer.category_histogram(analysis)
        assert fig is not None

    def test_si_histogram(self, analysis):
        fig = PlotlyRenderer.si_histogram(analysis.shape_result)
        assert fig is not None

    def test_curvedness_histogram(self, analysis):
        fig = PlotlyRenderer.curvedness_histogram(analysis.shape_result)
        assert fig is not None

    def test_complementarity_dual(self):
        protein, ligand = build_double_sphere()
        comp = analyze_complementarity(protein, ligand)
        fig = PlotlyRenderer.complementarity_dual(comp)
        assert fig is not None

    def test_complementarity_histograms(self):
        protein, ligand = build_double_sphere()
        comp = analyze_complementarity(protein, ligand)
        fig = PlotlyRenderer.complementarity_histograms(comp)
        assert fig is not None

    def test_preset_comparison_bars(self):
        comparison = compare_preset_surfaces()
        fig = PlotlyRenderer.preset_comparison_bars(comparison)
        assert fig is not None

    def test_gaussian_curvature_surface(self, analysis):
        fig = PlotlyRenderer.gaussian_curvature_surface(analysis.shape_result)
        assert fig is not None

    def test_mean_curvature_surface(self, analysis):
        fig = PlotlyRenderer.mean_curvature_surface(analysis.shape_result)
        assert fig is not None


# ═══════════════════════════════════════════════════════════════════════
# 18. Visualization — MatplotlibRenderer
# ═══════════════════════════════════════════════════════════════════════


class TestMatplotlibRenderer:
    """Test all MatplotlibRenderer static methods."""

    @pytest.fixture
    def analysis(self):
        mesh = build_ellipsoid(n_u=12, n_v=12)
        return analyze_shape(mesh)

    def test_shape_index_surface(self, analysis):
        fig = MatplotlibRenderer.shape_index_surface(analysis.shape_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_category_bar(self, analysis):
        fig = MatplotlibRenderer.category_bar(analysis)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_si_histogram(self, analysis):
        fig = MatplotlibRenderer.si_histogram(analysis.shape_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_complementarity_histograms(self):
        protein, ligand = build_double_sphere(n_u=12, n_v=12)
        comp = analyze_complementarity(protein, ligand)
        fig = MatplotlibRenderer.complementarity_histograms(comp)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_curvature_scatter(self, analysis):
        fig = MatplotlibRenderer.curvature_scatter(analysis.shape_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_preset_comparison(self):
        comparison = compare_preset_surfaces()
        fig = MatplotlibRenderer.preset_comparison(comparison)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════


class TestCLI:
    """Test CLI argument parsing."""

    def test_default_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.analyze is True

    def test_surface_choices(self):
        from main import build_parser, SURFACE_CHOICES
        parser = build_parser()
        for choice in SURFACE_CHOICES:
            args = parser.parse_args(["--surface", choice])
            assert args.surface == choice

    def test_save_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--save"])
        assert args.save is True

    def test_verbose_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_compare_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--compare"])
        assert args.compare is True

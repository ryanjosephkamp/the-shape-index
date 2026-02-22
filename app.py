"""
The Shape Index — Streamlit Application.

Six-page interactive app exploring Koenderink's Shape Index,
differential geometry of protein surfaces, patch segmentation,
saddle-point detection, and lock-and-key complementarity:

    1. Home & Overview
    2. The Shape Map (interactive 3-D Shape Index surface)
    3. The Topological Map (patch segmentation & saddle points)
    4. The Complementarity Test (protein vs ligand lock-and-key)
    5. Surface Comparison (preset surfaces side by side)
    6. Theory & Mathematics
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict

import numpy as np
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.shape_engine import (
    TriMesh,
    get_preset_surfaces,
    build_double_sphere,
    build_binding_pocket,
    build_bump,
    build_sphere,
    build_ellipsoid,
    build_saddle,
    build_torus,
    build_wavy_surface,
    compute_full_shape_index,
    segment_patches,
    detect_saddle_points,
    complementarity_analysis,
    SHAPE_CATEGORIES,
    SHAPE_CATEGORY_COLORS,
    DEFAULT_PATCH_BINS,
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
# Cached helpers
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def _cached_analysis(surface_name: str) -> FullShapeAnalysis:
    mesh = _SURFACE_MAP[surface_name]()
    return analyze_shape(mesh)


@st.cache_data(show_spinner=False)
def _cached_complementarity(mode: str) -> ComplementarityAnalysis:
    if mode == "Sphere Pair":
        protein, ligand = build_double_sphere()
    else:
        protein = build_binding_pocket()
        ligand = build_bump()
    return analyze_complementarity(protein, ligand)


@st.cache_data(show_spinner=False)
def _cached_preset_comparison() -> PresetComparisonResult:
    return compare_preset_surfaces()


# ═══════════════════════════════════════════════════════════════════════
# App configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Shape Index",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "🏠 Home",
    "🔷 The Shape Map",
    "🗺️ The Topological Map",
    "🔓 The Complementarity Test",
    "📊 Surface Comparison",
    "📚 Theory & Mathematics",
]

FOOTER = """
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <b>The Shape Index</b> – Week 14 Project 2 | Biophysics Portfolio<br>
    Ryan Kamp | University of Cincinnati Department of Computer Science<br>
    <a href="mailto:kamprj@mail.uc.edu">kamprj@mail.uc.edu</a> |
    <a href="https://github.com/ryanjosephkamp/the-shape-index">GitHub</a><br>
    February 21, 2026
</div>
"""

_SURFACE_MAP: Dict[str, callable] = {
    "Sphere": lambda: build_sphere(radius=1.0, name="Sphere"),
    "Ellipsoid": lambda: build_ellipsoid(a=2.0, b=1.0, c=0.5, name="Ellipsoid"),
    "Saddle": lambda: build_saddle(scale=2.0, name="Saddle (Hyperbolic Paraboloid)"),
    "Torus": lambda: build_torus(R=2.0, r=0.6, name="Torus"),
    "Wavy Surface": lambda: build_wavy_surface(name="Wavy Surface"),
    "Binding Pocket": lambda: build_binding_pocket(name="Binding Pocket"),
}

_SURFACE_OPTIONS = list(_SURFACE_MAP.keys())


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    st.sidebar.title("🔷 The Shape Index")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "*Week 14, Project 2 — Biophysics Portfolio*\n\n"
        "*Ryan Kamp • University of Cincinnati*"
    )
    return page


def _init_shared_state() -> None:
    """Ensure shared sidebar keys exist in session_state."""
    defaults = {
        "shared_surface": "Ellipsoid",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _surface_sidebar() -> dict:
    """Sidebar controls for surface selection."""
    _init_shared_state()

    st.sidebar.markdown("### Surface Parameters")

    surface = st.sidebar.selectbox(
        "Surface",
        _SURFACE_OPTIONS,
        key="shared_surface",
        help="Choose a preset surface to analyze.",
    )

    return {"surface": surface}


# ═══════════════════════════════════════════════════════════════════════
# Page 1 — Home
# ═══════════════════════════════════════════════════════════════════════


def page_home() -> None:
    """Render the landing page."""
    st.title("🔷 The Shape Index")
    st.subheader(
        "Differential Geometry of Binding — "
        "From Principal Curvatures to Surface Fingerprints"
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
Binding pockets aren't just "holes." They have specific shapes:
**Cup, Trough, Rut, Saddle, Ridge, Dome, Cap.**

This project implements **Koenderink's Shape Index**, a mathematical
transformation that maps the principal curvatures $(\\kappa_1, \\kappa_2)$
into a single **shape value** from −1 to +1:

$$S = \\frac{2}{\\pi} \\arctan\\!\\left(\\frac{\\kappa_1 + \\kappa_2}{\\kappa_1 - \\kappa_2}\\right)$$

- **S = −1** → **Cup** (spherical concavity)
- **S = 0** → **Saddle** (hyperbolic)
- **S = +1** → **Cap** (spherical convexity)

The Shape Index is **invariant to rotation** — it depends only on the
intrinsic geometry of the surface, making it an ideal descriptor for
computational drug design and protein–ligand complementarity analysis.
        """)

    with col2:
        st.markdown("""
**Key Features:**
- 🔷 Interactive 3-D Shape Index maps
- 🗺️ Topological patch segmentation
- 🔍 Saddle-point detection
- 🔓 Lock-and-Key complementarity test
- 📊 Six preset surfaces compared
- 📚 Full mathematical derivations
        """)

    st.markdown("---")

    # Quick preview — ellipsoid Shape Index
    with st.spinner("Computing Shape Index preview..."):
        analysis = _cached_analysis("Ellipsoid")
    fig_preview = PlotlyRenderer.shape_index_surface(
        analysis.shape_result,
        title="Preview — Ellipsoid Shape Index",
    )
    st.plotly_chart(fig_preview, use_container_width=True)

    with st.expander("ℹ️ About This Preview"):
        st.markdown("""
This ellipsoid preview demonstrates the Shape Index colour mapping:

- **Blue** = concave (Cup/Trough) — the elongated tips curve
  concavely in one direction
- **Grey** = saddle regions — where curvature changes sign
- **Red** = convex (Cap/Dome) — the broad sides curve convexly

The ellipsoid is a good test case because its three unequal
semi-axes (a=2, b=1, c=0.5) produce varied curvature across
the surface.
        """)

    st.markdown("---")

    st.markdown("### Quick Start")
    st.markdown("""
| Page | What You'll Find |
|------|------------------|
| 🔷 **The Shape Map** | Interactive 3-D visualization of Shape Index, Curvedness, Gaussian & Mean curvature |
| 🗺️ **The Topological Map** | Surface segmented into connected patches, saddle-point detection, hinge-region identification |
| 🔓 **The Complementarity Test** | Protein–Ligand lock-and-key test: concave pockets blue, convex ligands red |
| 📊 **Surface Comparison** | All six preset surfaces analysed side by side |
| 📚 **Theory & Mathematics** | Full derivations: principal curvatures, Shape Index, Curvedness, Gauss–Bonnet |
    """)

    with st.expander("ℹ️ About the Quick Start table"):
        st.markdown("""
Each page is designed to build understanding progressively:

1. **The Shape Map** shows you *what* the Shape Index looks like.
2. **The Topological Map** shows *why* it matters — real surfaces
   have discrete patches of different shape types.
3. **The Complementarity Test** shows the *application* — how
   shape complementarity validates the Lock-and-Key model.
4. **Surface Comparison** lets you compare *different geometries*.
5. **Theory** gives the full mathematical framework.
        """)

    st.markdown("---")

    st.markdown("### The Science Behind It")

    with st.expander("What is the Shape Index?"):
        st.markdown("""
The **Shape Index** was introduced by Koenderink & van Doorn (1992)
as a continuous measure of local surface shape. It maps the two
principal curvatures $(\\kappa_1, \\kappa_2)$ — which describe how
a surface curves in its two "most extreme" directions — into a
single number between −1 and +1.

$$S = \\frac{2}{\\pi} \\arctan\\!\\left(\\frac{\\kappa_1 + \\kappa_2}{\\kappa_1 - \\kappa_2}\\right)$$

**Why arctan?** The ratio $(\\kappa_1 + \\kappa_2) / (\\kappa_1 - \\kappa_2)$
captures the *relative* amounts of convexity vs. concavity. The
arctan maps this ratio to a bounded range, and the $2/\\pi$ factor
normalises it to $[-1, +1]$.

**Key property:** The Shape Index is independent of the *magnitude*
of curvature — it depends only on the *type* of shape. A tiny cup
and a giant cup both have $S = -1$.
        """)

    with st.expander("What is Curvedness?"):
        st.markdown("""
While the Shape Index captures *type*, **Curvedness** captures
*magnitude*:

$$C = \\sqrt{\\frac{\\kappa_1^2 + \\kappa_2^2}{2}}$$

- A flat plane has $C = 0$.
- A tight sphere has large $C$.
- A gently curved surface has small $C$.

Together, $(S, C)$ fully characterise the local shape: $S$ tells
you *what kind* of shape, and $C$ tells you *how much* shape.
        """)

    with st.expander("Why Does Shape Complementarity Matter?"):
        st.markdown("""
**The Lock-and-Key hypothesis** (Emil Fischer, 1894) states that an
enzyme's active site is geometrically complementary to its substrate.

In modern computational terms:
- The **protein binding site** should be predominantly **concave**
  (Cup/Trough, $S < -0.25$).
- The **ligand contact surface** should be predominantly **convex**
  (Cap/Dome, $S > 0.25$).

Shape complementarity scores are used in:
- Molecular docking (scoring functions)
- Virtual screening (binding site prediction)
- De novo drug design (shape-based generation)
- Protein–protein interaction prediction
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 2 — The Shape Map
# ═══════════════════════════════════════════════════════════════════════


def page_shape_map() -> None:
    """Interactive 3-D Shape Index visualisation."""
    st.title("🔷 The Shape Map")
    st.markdown(
        "Explore the differential geometry of surfaces in 3-D. "
        "Each vertex is coloured by its **Shape Index** — from "
        "**blue (Cup, S=−1)** through **grey (Saddle, S=0)** to "
        "**red (Cap, S=+1)**."
    )
    st.markdown("---")

    params = _surface_sidebar()

    with st.spinner("Computing Shape Index..."):
        analysis = _cached_analysis(params["surface"])

    result = analysis.shape_result

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Shape Index", f"{analysis.mean_shape_index:+.3f}")
    col2.metric("Std Shape Index", f"{analysis.std_shape_index:.3f}")
    col3.metric("Mean Curvedness", f"{analysis.mean_curvedness:.3f}")
    col4.metric("Vertices", f"{analysis.mesh.n_vertices:,}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Mean Shape Index:** Average over all vertices. Negative means
  the surface is predominantly concave; positive means convex;
  near zero means mixed or saddle-like.
- **Std Shape Index:** How varied the shape is across the surface.
  A sphere has zero variance (all vertices have the same SI).
- **Mean Curvedness:** How curved the surface is on average.
  Flat surfaces have low C; tightly curved surfaces have high C.
- **Vertices:** Number of mesh vertices used in the analysis.
        """)

    st.markdown("---")

    # ── 3-D Shape Index Surface ──
    st.markdown("### Shape Index Surface")
    fig_si = PlotlyRenderer.shape_index_surface(
        result,
        title=f"Shape Index — {analysis.mesh.name}",
    )
    st.plotly_chart(fig_si, use_container_width=True)

    with st.expander("ℹ️ Reading the Shape Index Surface"):
        st.markdown("""
**Colour scale:**
- 🔵 **Blue (S ≈ −1):** Cup — spherical concavity. Think of the
  inside of a bowl.
- 🔵 **Light blue (S ≈ −0.5):** Rut/Trough — elongated concavity.
  Think of a gutter.
- ⚪ **Grey (S ≈ 0):** Saddle — curvature changes sign. The
  surface curves up in one direction and down in the other.
- 🟠 **Orange (S ≈ +0.5):** Ridge/Dome — elongated convexity.
  Think of the top of a hill.
- 🔴 **Red (S ≈ +1):** Cap — spherical convexity. Think of the
  outside of a ball.

**Interaction tips:**
- Rotate by clicking and dragging.
- Zoom with the scroll wheel.
- Hover to see exact (x, y, z) and SI values.
        """)

    st.markdown("---")

    # ── Curvedness Surface ──
    st.markdown("### Curvedness Surface")
    fig_c = PlotlyRenderer.curvedness_surface(
        result,
        title=f"Curvedness — {analysis.mesh.name}",
    )
    st.plotly_chart(fig_c, use_container_width=True)

    with st.expander("ℹ️ Reading the Curvedness Surface"):
        st.markdown("""
**Curvedness** $C = \\sqrt{(\\kappa_1^2 + \\kappa_2^2)/2}$ measures
the total magnitude of bending at each vertex.

- **Dark (low C):** Nearly flat or very gently curved.
- **Bright (high C):** Tightly curved — sharp features.

Curvedness combined with Shape Index gives a complete local
shape descriptor: SI says *what type* (cup, saddle, cap);
C says *how strongly*.
        """)

    st.markdown("---")

    # ── Gaussian & Mean Curvature ──
    st.markdown("### Gaussian & Mean Curvature")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_K = PlotlyRenderer.gaussian_curvature_surface(
            result,
            title=f"Gaussian Curvature K — {analysis.mesh.name}",
        )
        st.plotly_chart(fig_K, use_container_width=True)

    with col_b:
        fig_H = PlotlyRenderer.mean_curvature_surface(
            result,
            title=f"Mean Curvature H — {analysis.mesh.name}",
        )
        st.plotly_chart(fig_H, use_container_width=True)

    with st.expander("ℹ️ Gaussian vs Mean Curvature"):
        st.markdown("""
**Gaussian curvature** $K = \\kappa_1 \\cdot \\kappa_2$:
- $K > 0$: elliptic point (both curvatures same sign — dome or bowl)
- $K = 0$: parabolic point (one curvature is zero — cylinder-like)
- $K < 0$: hyperbolic point (curvatures opposite sign — saddle)

**Mean curvature** $H = (\\kappa_1 + \\kappa_2) / 2$:
- $H > 0$: predominantly convex
- $H = 0$: minimal surface (soap film)
- $H < 0$: predominantly concave

The **Gauss–Bonnet theorem** relates the total Gaussian curvature
to the topology: $\\int K \\, dA = 2\\pi \\chi$, where $\\chi$ is the
Euler characteristic.
        """)

    st.markdown("---")

    # ── Category Distribution ──
    st.markdown("### Shape Category Distribution")
    col_c, col_d = st.columns(2)

    with col_c:
        fig_cat = PlotlyRenderer.category_histogram(analysis)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_d:
        fig_hist = PlotlyRenderer.si_histogram(result)
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("ℹ️ Interpreting the Distributions"):
        st.markdown("""
**Category bar chart (left):** Each vertex is classified into one
of nine canonical shape categories based on its Shape Index.
A sphere should be nearly all "Cap"; a saddle surface should be
nearly all "Saddle."

**Continuous histogram (right):** Shows the full distribution of
Shape Index values. Peaks indicate dominant shape types; the width
shows shape diversity.

**Nine Koenderink categories:**

| S range | Category | Description |
|---------|----------|-------------|
| [−1.00, −0.75) | Cup | Spherical concavity |
| [−0.75, −0.50) | Trough | Deep elongated concavity |
| [−0.50, −0.25) | Rut | Shallow elongated concavity |
| [−0.25, −0.05) | Saddle Rut | Nearly saddle, slightly concave |
| [−0.05, +0.05) | Saddle | Pure saddle (hyperbolic) |
| [+0.05, +0.25) | Saddle Ridge | Nearly saddle, slightly convex |
| [+0.25, +0.50) | Ridge | Shallow elongated convexity |
| [+0.50, +0.75) | Dome | Deep elongated convexity |
| [+0.75, +1.00] | Cap | Spherical convexity |
        """)

    st.markdown("---")

    # ── Principal Curvature Scatter ──
    st.markdown("### Principal Curvature Space (κ₁ vs κ₂)")
    st.markdown(
        "Each point is a mesh vertex plotted at its principal "
        "curvatures $(\\kappa_2, \\kappa_1)$, coloured by Shape Index. "
        "The diagonal $\\kappa_1 = \\kappa_2$ represents umbilical "
        "points (spherical)."
    )

    fig_scatter = MatplotlibRenderer.curvature_scatter(result)
    st.pyplot(fig_scatter)

    with st.expander("ℹ️ Reading the Curvature Scatter Plot"):
        st.markdown("""
- **Points on the diagonal** ($\\kappa_1 = \\kappa_2$): Umbilical
  points — locally spherical. Above the diagonal: Cap ($S = +1$);
  below: Cup ($S = -1$).
- **Points on the axes** (one $\\kappa = 0$): Parabolic points —
  locally cylindrical.
- **Points in different quadrants:**
  - Top-right: both curvatures positive → convex (Cap/Dome)
  - Bottom-left: both negative → concave (Cup/Trough)
  - Other quadrants: opposite signs → saddle-like
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 3 — The Topological Map
# ═══════════════════════════════════════════════════════════════════════


def page_topological_map() -> None:
    """Patch segmentation and saddle-point detection."""
    st.title("🗺️ The Topological Map")
    st.markdown(
        "The surface is segmented into **connected patches** of "
        "similar Shape Index. **Saddle points** are highlighted — "
        "in enzymology, these often indicate dynamic hinge regions "
        "or transition-state stabilisation sites."
    )
    st.markdown("---")

    params = _surface_sidebar()

    with st.spinner("Computing patches..."):
        analysis = _cached_analysis(params["surface"])

    patches = analysis.patches
    saddles = analysis.saddle_points
    patch_stats = analyze_patches(analysis)
    saddle_cat = analyze_saddle_points(analysis)

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patches", f"{patches.n_patches}")
    col2.metric("Saddle Points", f"{len(saddles)}")
    col3.metric("Saddle Fraction", f"{saddle_cat.fraction_saddle:.1%}")
    col4.metric("Spatial Clusters", f"{len(saddle_cat.spatial_clusters)}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Total Patches:** Number of connected regions where vertices
  share a similar Shape Index bin. More patches = more shape
  diversity.
- **Saddle Points:** Vertices where |S| < 0.10 (near-zero Shape
  Index). These are hyperbolic points where the surface changes
  from convex to concave.
- **Saddle Fraction:** What proportion of all vertices are saddle
  points. Higher values indicate more complex topology.
- **Spatial Clusters:** Saddle points grouped by proximity (within
  1.0 units). Each cluster may represent a functional region.
        """)

    st.markdown("---")

    # ── Patch Map ──
    st.markdown("### Patch Map")
    fig_patches = PlotlyRenderer.patch_map(
        analysis.shape_result, patches,
        title=f"Surface Patches — {analysis.mesh.name}",
    )
    st.plotly_chart(fig_patches, use_container_width=True)

    with st.expander("ℹ️ Reading the Patch Map"):
        st.markdown("""
Each colour represents a distinct connected patch — a region of
the surface where all vertices have similar Shape Index values.

**How patches are computed:**
1. **Bin** each vertex's Shape Index into 9 equal bands (matching
   the nine Koenderink categories).
2. **Find connected components** within each bin using BFS on the
   mesh adjacency graph.
3. **Label** each component as a patch and compute its area,
   centroid, and dominant shape category.

**Why patches matter:**
- In protein surface analysis, patches identify functional regions:
  a large Cup patch might be a binding pocket; a Cap patch might
  be a protruding loop.
- Patch size distribution reveals surface complexity.
        """)

    st.markdown("---")

    # ── Saddle-Point Overlay ──
    st.markdown("### Saddle-Point Overlay")
    fig_saddle = PlotlyRenderer.saddle_overlay(
        analysis.shape_result, saddles,
        title=f"Saddle Points — {analysis.mesh.name}",
    )
    st.plotly_chart(fig_saddle, use_container_width=True)

    with st.expander("ℹ️ Why Saddle Points Matter"):
        st.markdown("""
**Saddle points** are where the Shape Index crosses zero — the
surface curves up in one direction and down in the other.

**In enzymology and structural biology:**
- **Hinge regions:** Flexible loops connecting rigid domains often
  sit at saddle points on the molecular surface.
- **Transition-state stabilisation:** Enzymatic active sites can
  have saddle-like geometry that complements the geometry of the
  transition state.
- **Channel entrances:** Tunnels and channels in proteins begin
  at saddle points where the surface topology changes.

**Yellow diamonds** mark detected saddle vertices (|S| < 0.10).
Their **curvedness** indicates how sharply the surface bends at
these transition points.
        """)

    st.markdown("---")

    # ── Patch Statistics ──
    st.markdown("### Patch Statistics")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Category Patch Counts**")
        import pandas as pd
        cat_df = pd.DataFrame(
            [(cat, cnt) for cat, cnt in patch_stats.category_patch_counts.items()
             if cnt > 0],
            columns=["Category", "Patches"],
        ).sort_values("Patches", ascending=False)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Patch Area Statistics**")
        st.markdown(f"""
| Statistic | Value |
|-----------|-------|
| Total patches | {patch_stats.n_patches} |
| Mean area | {patch_stats.mean_patch_area:.4f} |
| Std area | {patch_stats.std_patch_area:.4f} |
| Largest patch | #{patch_stats.largest_patch_id} |
| Largest area | {patch_stats.largest_patch_area:.4f} |
| Largest category | {patch_stats.largest_patch_category} |
        """)

    with st.expander("ℹ️ Interpreting Patch Statistics"):
        st.markdown("""
**Category patch counts** show how many distinct patches each shape
type produces. A surface with many Saddle patches is topologically
complex; one with a single Cap patch is simple (like a sphere).

**Area statistics** reveal the size distribution. In protein analysis:
- Large Cup patches → binding pockets
- Many small Saddle patches → surface roughness
- Uniform patch sizes → regular geometry (sphere, cylinder)
        """)

    st.markdown("---")

    # ── Saddle Catalogue ──
    st.markdown("### Saddle-Point Catalogue")
    st.markdown(saddle_cat.explanation)

    if saddle_cat.n_saddle > 0:
        saddle_df = pd.DataFrame([
            {
                "Vertex": sp.vertex_index,
                "x": f"{sp.position[0]:.3f}",
                "y": f"{sp.position[1]:.3f}",
                "z": f"{sp.position[2]:.3f}",
                "SI": f"{sp.shape_index:+.4f}",
                "C": f"{sp.curvedness:.4f}",
                "Patch": sp.patch_id,
            }
            for sp in saddle_cat.saddle_points[:50]
        ])
        st.dataframe(saddle_df, use_container_width=True, hide_index=True)
        if saddle_cat.n_saddle > 50:
            st.caption(f"Showing first 50 of {saddle_cat.n_saddle} saddle points.")

    with st.expander("ℹ️ Reading the Saddle-Point Table"):
        st.markdown("""
Each row is a vertex with near-zero Shape Index (|S| < 0.10):
- **Vertex:** Index in the mesh.
- **(x, y, z):** 3-D coordinates of the saddle point.
- **SI:** Shape Index (should be near zero).
- **C:** Curvedness — how sharply the surface bends at this saddle.
  High C saddles are "sharp" transitions; low C saddles are gentle.
- **Patch:** Which patch this vertex belongs to.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 4 — The Complementarity Test
# ═══════════════════════════════════════════════════════════════════════


def page_complementarity() -> None:
    """Lock-and-Key complementarity test."""
    st.title("🔓 The Complementarity Test")
    st.markdown(
        "**The Lock-and-Key theory** predicts that the protein "
        "binding site (concave, **blue**) complements the ligand "
        "surface (convex, **red**). Load a protein–ligand pair, "
        '"open the book" (separate them), and see the Shape Index '
        "prove complementarity."
    )
    st.markdown("---")

    st.sidebar.markdown("### Complementarity Mode")
    mode = st.sidebar.radio(
        "Test pair",
        ["Sphere Pair", "Pocket + Bump"],
        help="Choose the protein–ligand mock system.",
    )

    with st.spinner("Running complementarity analysis..."):
        comp = _cached_complementarity(mode)

    cr = comp.comp_result

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Complementarity Score", f"{cr.complementarity_score:.3f}")
    col2.metric("Protein Cup%", f"{cr.protein_cup_fraction:.1%}")
    col3.metric("Ligand Cap%", f"{cr.ligand_cap_fraction:.1%}")
    col4.metric("Mirror Score", f"{comp.mirror_score:.3f}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Complementarity Score:** Average of protein concave fraction
  and ligand convex fraction. Ranges from 0 (no match) to 1
  (perfect lock-and-key). Values above 0.6 indicate strong
  complementarity.
- **Protein Cup%:** Fraction of protein surface vertices with
  S < −0.25 (concave: Cup, Trough, Rut). Should be high for a
  good binding pocket.
- **Ligand Cap%:** Fraction of ligand surface vertices with
  S > 0.25 (convex: Cap, Dome, Ridge). Should be high for a
  protruding ligand.
- **Mirror Score:** How well the protein SI histogram mirrors the
  ligand SI histogram when flipped around S=0. Higher = better
  complementarity.
        """)

    st.markdown("---")

    # ── Dual Surface ──
    st.markdown("### Protein vs Ligand — Shape Index Surfaces")
    st.markdown(
        "The **protein** surface (left) should be predominantly "
        "**blue (concave)**, and the **ligand** surface (right) "
        "should be predominantly **red (convex)**."
    )
    fig_dual = PlotlyRenderer.complementarity_dual(comp)
    st.plotly_chart(fig_dual, use_container_width=True)

    with st.expander("ℹ️ Reading the Dual Surface View"):
        st.markdown("""
This is the "open the book" view. Imagine the protein and ligand
were bound together, and you've pulled them apart:

- **Left (Protein):** The binding interface should show concave
  (blue) regions — this is the "lock."
- **Right (Ligand):** The contact surface should show convex
  (red) regions — this is the "key."

**Perfect complementarity** means the blue regions on the protein
exactly match the red regions on the ligand in shape and position.

**Rotate each surface** to explore the shape from different angles.
        """)

    st.markdown("---")

    # ── Histogram Overlay ──
    st.markdown("### Shape Index Histograms")
    fig_hist = PlotlyRenderer.complementarity_histograms(comp)
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("ℹ️ Reading the Histogram Overlay"):
        st.markdown("""
**Blue bars** = protein SI distribution; **red bars** = ligand SI
distribution.

**Perfect complementarity** shows:
- Protein histogram peaks in the **negative** (concave) region
- Ligand histogram peaks in the **positive** (convex) region
- The two distributions are roughly **mirror images** around S=0

**What to look for:**
- Clear separation: protein mostly on the left, ligand on the right
- Similar widths: the concavity and convexity are on the same scale
- The **complementarity score** quantifies this separation
        """)

    st.markdown("---")

    # ── Detailed Analysis ──
    st.markdown("### Detailed Complementarity Analysis")
    st.markdown(comp.explanation)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Protein Category Fractions**")
        fig_prot_cat = PlotlyRenderer.category_histogram(comp.protein_analysis)
        st.plotly_chart(fig_prot_cat, use_container_width=True)

    with col_b:
        st.markdown("**Ligand Category Fractions**")
        fig_lig_cat = PlotlyRenderer.category_histogram(comp.ligand_analysis)
        st.plotly_chart(fig_lig_cat, use_container_width=True)

    with st.expander("ℹ️ Category Breakdown"):
        st.markdown("""
Comparing the category distributions confirms the lock-and-key
relationship:

- **Protein** should have high Cup/Trough/Rut fractions.
- **Ligand** should have high Cap/Dome/Ridge fractions.
- **Saddle** fractions on both sides represent transition regions
  between the complementary shapes.

In real protein–ligand complexes, the complementarity is never
perfect (induced-fit effects, water-mediated contacts, etc.),
but the Shape Index still captures the dominant geometric trend.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 5 — Surface Comparison
# ═══════════════════════════════════════════════════════════════════════


def page_surface_comparison() -> None:
    """Compare all preset surfaces."""
    st.title("📊 Surface Comparison")
    st.markdown(
        "All six preset surfaces analysed side by side: Sphere, "
        "Ellipsoid, Saddle, Torus, Wavy Surface, Binding Pocket."
    )
    st.markdown("---")

    with st.spinner("Analysing all surfaces..."):
        comparison = _cached_preset_comparison()

    # ── Summary Table ──
    st.markdown("### Summary Table")
    import pandas as pd
    df = pd.DataFrame(comparison.summary_table)
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Reading the Summary Table"):
        st.markdown("""
| Column | Meaning |
|--------|---------|
| Surface | Preset surface name |
| Vertices | Number of mesh vertices |
| Faces | Number of triangular faces |
| Area | Total surface area |
| Mean SI | Mean Shape Index (negative=concave, positive=convex) |
| Std SI | Standard deviation of Shape Index |
| Mean C | Mean Curvedness (higher=more curved) |
| Dominant | Most common shape category |
| Patches | Number of connected patches |
| Saddles | Number of saddle-point vertices |
        """)

    st.markdown("---")

    # ── Grouped Bar Chart ──
    st.markdown("### Mean Shape Index & Curvedness")
    fig_bars = PlotlyRenderer.preset_comparison_bars(comparison)
    st.plotly_chart(fig_bars, use_container_width=True)

    with st.expander("ℹ️ Interpreting the Bar Chart"):
        st.markdown("""
- **Shape Index bars** show whether each surface is predominantly
  convex (positive), concave (negative), or mixed (near zero).
- **Curvedness bars** show how sharply each surface curves.

**Expected patterns:**
- **Sphere:** Mean SI near +1 (all Cap), moderate C.
- **Ellipsoid:** Mean SI mildly positive, moderate C.
- **Saddle:** Mean SI near 0, moderate C.
- **Torus:** Mean SI mixed (inner concave, outer convex).
- **Wavy Surface:** Mean SI near 0 (balanced), variable C.
- **Binding Pocket:** Mean SI mildly negative (concave center).
        """)

    st.markdown("---")

    # ── Per-Surface Shape Index Maps ──
    st.markdown("### Shape Index Maps")
    surface_name = st.selectbox(
        "Select surface to view",
        list(comparison.analyses.keys()),
        help="Choose a surface to visualise in 3-D.",
    )
    an = comparison.analyses[surface_name]
    fig_si = PlotlyRenderer.shape_index_surface(
        an.shape_result,
        title=f"Shape Index — {surface_name}",
    )
    st.plotly_chart(fig_si, use_container_width=True)

    with st.expander("ℹ️ About This Surface"):
        st.markdown(an.explanation)

    st.markdown("---")

    # ── Category Comparison ──
    st.markdown("### Category Distributions")
    cols = st.columns(3)
    surface_names = list(comparison.analyses.keys())
    for i, name in enumerate(surface_names):
        with cols[i % 3]:
            fig_cat = PlotlyRenderer.category_histogram(
                comparison.analyses[name],
                title=name,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("ℹ️ Comparing Category Distributions"):
        st.markdown("""
These charts reveal each surface's shape "fingerprint":

- **Sphere:** Nearly all Cap — uniform convexity.
- **Ellipsoid:** Mix of Cap, Dome, Ridge — varying convexity due
  to different semi-axes.
- **Saddle:** Mostly Saddle — perfect hyperbolic geometry.
- **Torus:** Bimodal — outer ring is convex (Cap/Dome), inner
  ring is concave (Cup/Trough). The transition produces saddle.
- **Wavy Surface:** Balanced mix — alternating hills and valleys.
- **Binding Pocket:** Centre is concave (Cup); perimeter is flat
  (near-Saddle).
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 6 — Theory & Mathematics
# ═══════════════════════════════════════════════════════════════════════


def page_theory() -> None:
    """Full mathematical derivations."""
    st.title("📚 Theory & Mathematics")
    st.markdown(
        "Complete mathematical framework for differential geometry "
        "of surfaces, principal curvatures, Koenderink's Shape Index, "
        "patch segmentation, and shape complementarity."
    )
    st.markdown("---")

    # ── Section 1: Principal Curvatures ──
    with st.expander("1. Principal Curvatures"):
        st.markdown(r"""
### Principal Curvatures

At each point on a smooth surface, the **normal curvature** $\kappa_n$
varies with direction. The two extreme values are the **principal
curvatures**:

$$\kappa_1 = \max_\theta \kappa_n(\theta), \quad \kappa_2 = \min_\theta \kappa_n(\theta)$$

where $\theta$ parameterises the direction in the tangent plane.

**Convention:** $\kappa_1 \geq \kappa_2$.

The principal curvatures are eigenvalues of the **shape operator**
(Weingarten map):

$$\mathbf{S} = -d\mathbf{n} \cdot d\mathbf{r}^{-1}$$

In a local tangent frame $(e_1, e_2)$, the shape operator becomes
a 2×2 symmetric matrix. Its eigenvalues are $(\kappa_1, \kappa_2)$
and its eigenvectors are the **principal directions**.

#### Discrete Estimation

On a triangle mesh, we estimate the shape operator at each vertex
by fitting a local quadratic patch to the 1-ring neighbourhood:

$$h(u, v) \approx a u^2 + b u v + c v^2$$

The shape operator in the local frame is then:

$$\mathbf{S} = \begin{pmatrix} 2a & b \\ b & 2c \end{pmatrix}$$

Eigenvalues of this matrix give $(\kappa_1, \kappa_2)$.
        """)

    # ── Section 2: Koenderink's Shape Index ──
    with st.expander("2. Koenderink's Shape Index"):
        st.markdown(r"""
### Shape Index

**Definition** (Koenderink & van Doorn, 1992):

$$S = \frac{2}{\pi} \arctan\!\left(\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right)$$

**Domain:** $S \in [-1, +1]$.

**Physical interpretation:**

| S value | Shape | Description |
|---------|-------|-------------|
| $-1$ | Cup | Spherical concavity |
| $-0.5$ | Rut | Elongated concavity |
| $0$ | Saddle | Hyperbolic point |
| $+0.5$ | Ridge | Elongated convexity |
| $+1$ | Cap | Spherical convexity |

**Why this formula works:**

The ratio $(\kappa_1 + \kappa_2)/(\kappa_1 - \kappa_2)$ captures
the balance between the sum (indicating overall
convexity/concavity) and the difference (indicating anisotropy).
The $\arctan$ maps this to a bounded range, and $2/\pi$ normalises
to $[-1, +1]$.

**Key properties:**
- **Scale-invariant:** Scaling the surface by a factor $\lambda$
  scales both curvatures but leaves S unchanged.
- **Rotation-invariant:** S depends only on the eigenvalues of the
  shape operator, not their orientation.
- **Continuous:** S varies smoothly across the surface (wherever
  $\kappa_1 \neq \kappa_2$).
- **Undefined at umbilics:** When $\kappa_1 = \kappa_2$, the
  denominator is zero. We assign S = 0 for flat points and
  S = ±1 for spherical umbilics.
        """)

    # ── Section 3: Curvedness ──
    with st.expander("3. Curvedness"):
        st.markdown(r"""
### Curvedness

**Definition:**

$$C = \sqrt{\frac{\kappa_1^2 + \kappa_2^2}{2}}$$

While the Shape Index captures the *type* of curvature (cup vs. cap),
Curvedness captures the *magnitude* — how tightly curved the surface is.

**Properties:**
- $C = 0$ for a flat plane.
- $C = |\kappa|$ for a sphere with curvature $\kappa$.
- $C$ increases with the "sharpness" of features.

**Together**, $(S, C)$ form a complete local shape descriptor:
- $S$ says *what kind* of shape (cup, saddle, cap).
- $C$ says *how much* shape (flat, gently curved, sharply curved).

This decomposition is analogous to polar coordinates in curvature
space: $S$ is the "angle" and $C$ is the "radius."
        """)

    # ── Section 4: Gaussian and Mean Curvature ──
    with st.expander("4. Gaussian and Mean Curvature"):
        st.markdown(r"""
### Gaussian Curvature

$$K = \kappa_1 \cdot \kappa_2$$

**Classification of surface points:**
- $K > 0$: **Elliptic** — locally dome-shaped or bowl-shaped.
- $K = 0$: **Parabolic** — locally cylindrical.
- $K < 0$: **Hyperbolic** — locally saddle-shaped.

**Gauss–Bonnet theorem:** For a closed surface,

$$\int_S K \, dA = 2\pi \chi(S)$$

where $\chi$ is the **Euler characteristic**. For a sphere,
$\chi = 2$; for a torus, $\chi = 0$.

### Mean Curvature

$$H = \frac{\kappa_1 + \kappa_2}{2}$$

- $H > 0$: predominantly convex.
- $H = 0$: **minimal surface** (soap film).
- $H < 0$: predominantly concave.

**Relationship to Shape Index:**

$$S = \frac{2}{\pi} \arctan\!\left(\frac{2H}{\kappa_1 - \kappa_2}\right)$$
        """)

    # ── Section 5: Patch Segmentation ──
    with st.expander("5. Patch Segmentation Algorithm"):
        st.markdown(r"""
### Patch Segmentation

The surface is segmented into **connected patches** of similar shape:

**Algorithm:**
1. **Bin** each vertex's Shape Index into $n$ equal bands over
   $[-1, +1]$.
2. For each bin, **find connected components** on the mesh
   adjacency graph (BFS/DFS restricted to vertices in that bin).
3. **Label** each component as a patch.
4. Compute per-patch statistics: area, centroid, dominant category.

**Complexity:** $O(V + E)$ where $V$ = vertices, $E$ = edges.

**Applications in structural biology:**
- **Binding site identification:** Large concave patches (Cup/Trough)
  are candidate binding pockets.
- **Surface fingerprinting:** The distribution of patch sizes and
  types creates a rotation-invariant "fingerprint" for each protein.
- **Docking:** Shape-based docking aligns complementary patches
  between protein and ligand.
        """)

    # ── Section 6: Saddle-Point Detection ──
    with st.expander("6. Saddle-Point Detection"):
        st.markdown(r"""
### Saddle-Point Detection

A saddle point is where $S \approx 0$ — the surface curves up in
one direction and down in the other ($\kappa_1 > 0 > \kappa_2$).

**Detection:** Select vertices where $|S| < \epsilon$ (threshold,
typically 0.10).

**Biological significance:**
- **Hinge regions:** Saddle geometry at the junction between rigid
  domains allows the protein to flex.
- **Transition states:** Enzyme active sites may have saddle-like
  geometry that complements the transition state of the reaction.
- **Channel entrances:** The mouth of a tunnel or channel is
  typically a saddle point — the surface curves inward along
  the channel axis and outward perpendicular to it.

**Spatial clustering:** Nearby saddle points are grouped into
clusters to identify functional regions rather than individual
vertices.
        """)

    # ── Section 7: Shape Complementarity ──
    with st.expander("7. Shape Complementarity (Lock-and-Key)"):
        st.markdown(r"""
### Shape Complementarity

**Fischer's Lock-and-Key hypothesis** (1894): An enzyme's active
site is geometrically complementary to its substrate.

**Quantification via Shape Index:**
- The **protein binding site** should have $S < -0.25$ (concave:
  Cup, Trough, Rut).
- The **ligand contact surface** should have $S > +0.25$ (convex:
  Cap, Dome, Ridge).

**Complementarity score:**

$$\text{Score} = \frac{f_{\text{protein}}^{\text{concave}} + f_{\text{ligand}}^{\text{convex}}}{2}$$

where $f$ denotes the fraction of vertices in the specified range.

**Histogram mirror test:** If the protein and ligand are perfectly
complementary, flipping the ligand's SI histogram around $S = 0$
should produce a distribution similar to the protein's. The
correlation of these mirrored distributions gives the **mirror
score**.

**Interpretation:**
- Score > 0.6: Strong complementarity (lock-and-key).
- Score 0.4–0.6: Moderate (induced-fit may play a role).
- Score < 0.4: Low complementarity.
        """)

    # ── Section 8: The Shape Operator ──
    with st.expander("8. The Shape Operator (Weingarten Map)"):
        st.markdown(r"""
### The Shape Operator

The **shape operator** (or Weingarten map) $\mathbf{S}_p$ at a
point $p$ on a surface describes how the normal vector changes:

$$\mathbf{S}_p(\mathbf{v}) = -D_\mathbf{v}\mathbf{n}$$

In a local orthonormal tangent frame $(e_1, e_2)$, the shape
operator is represented by the matrix:

$$\mathbf{S} = \mathbf{I}^{-1}\mathbf{II}$$

where $\mathbf{I}$ is the first fundamental form (metric tensor)
and $\mathbf{II}$ is the second fundamental form.

For a height function $z = f(x, y)$ over a tangent plane:

$$\mathbf{S} = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{pmatrix} \cdot \frac{1}{\sqrt{1 + f_x^2 + f_y^2}}$$

At the origin (where the tangent plane is flat), this simplifies to
the **Hessian matrix** of the height function.
        """)

    # ── Section 9: Discrete Differential Geometry ──
    with st.expander("9. Discrete Differential Geometry on Meshes"):
        st.markdown(r"""
### Discrete Estimation

On a triangle mesh, smooth differential geometric quantities are
approximated from the discrete vertex and face data.

**Vertex normals:** Area-weighted average of adjacent face normals:

$$\mathbf{n}_v = \frac{\sum_f A_f \mathbf{n}_f}{\left\|\sum_f A_f \mathbf{n}_f\right\|}$$

**Local quadratic fitting:**
1. Choose a local tangent frame $(e_1, e_2, \mathbf{n})$ at vertex $v$.
2. Project 1-ring neighbours into the tangent frame: $(u_i, v_i, h_i)$.
3. Fit: $h = a u^2 + b u v + c v^2$ via least squares.
4. Extract shape operator: $\mathbf{S} = \begin{pmatrix}2a & b\\b & 2c\end{pmatrix}$.
5. Eigenvalues → $(\kappa_1, \kappa_2)$.

This approach is $O(V \cdot k)$ where $k$ is the average vertex
valence (typically 6 for triangle meshes).

**Alternative methods:**
- Cotangent Laplacian → mean curvature normal
- Angle-deficit → Gaussian curvature (Gauss–Bonnet)
- Jet fitting (higher-order polynomial fits)
        """)

    # ── Section 10: Applications in Drug Design ──
    with st.expander("10. Applications in Computational Drug Design"):
        st.markdown(r"""
### Shape Index in Drug Discovery

**Binding pocket detection:**
- Scan the protein surface for large connected patches with
  $S < -0.5$ (Cup/Trough).
- Rank pockets by area, curvedness, and topological complexity.

**Ligandability assessment:**
- "Druggable" pockets tend to have:
  - Well-defined concave patches (high cup fraction)
  - Moderate curvedness (not too flat, not too tight)
  - Some saddle features at the rim (complementarity to drug-like
    shapes)

**Shape matching in docking:**
- Align surfaces by matching complementary Shape Index patches.
- Score docking poses by the overlap of concave/convex regions.

**De novo drug design:**
- Generate molecular shapes that maximise complementarity score
  with a target pocket.
- The Shape Index provides a differentiable (in principle) surface
  descriptor for gradient-based optimisation.

**References:**
- Koenderink, J. J. & van Doorn, A. J. (1992). *Image Vis. Comput.*, 10(8).
- Connolly, M. L. (1986). *Biopolymers*, 25, 1229–1247.
- Lawrence, M. C. & Colman, P. M. (1993). *J. Mol. Biol.*, 234, 946–950.
        """)

    # ── Section 11: Mathematical Relations ──
    with st.expander("11. Mathematical Relations Summary"):
        st.markdown(r"""
### Key Relations

| Quantity | Formula |
|----------|---------|
| Principal curvatures | $\kappa_1, \kappa_2 = $ eigenvalues of $\mathbf{S}$ |
| Mean curvature | $H = (\kappa_1 + \kappa_2)/2$ |
| Gaussian curvature | $K = \kappa_1 \cdot \kappa_2$ |
| Shape Index | $S = (2/\pi)\arctan((\kappa_1+\kappa_2)/(\kappa_1-\kappa_2))$ |
| Curvedness | $C = \sqrt{(\kappa_1^2+\kappa_2^2)/2}$ |
| Curvature from (S,C) | $\kappa_1 = C(1+\sin(\pi S/2))$, $\kappa_2 = C(1-\sin(\pi S/2))$ |
| Gauss–Bonnet | $\int K\,dA = 2\pi\chi$ |
| Complementarity | $(f_{\text{prot}}^{<-0.25} + f_{\text{lig}}^{>0.25})/2$ |

### Inverse Relations

Given $(S, C)$, the principal curvatures can be recovered:

$$\kappa_1 = C\left(1 + \sin\!\frac{\pi S}{2}\right), \quad \kappa_2 = C\left(1 - \sin\!\frac{\pi S}{2}\right)$$

This shows that $(S, C)$ is indeed a complete parameterisation of
the curvature: they encode the same information as
$(\kappa_1, \kappa_2)$ but in a more interpretable form.
        """)

    # ── Section 12: References ──
    with st.expander("12. References"):
        st.markdown("""
1. Koenderink, J. J. & van Doorn, A. J. (1992). Surface shape and
   curvature scales. *Image and Vision Computing*, 10(8), 557–564.

2. Fischer, E. (1894). Einfluss der Configuration auf die Wirkung
   der Enzyme. *Berichte der deutschen chemischen Gesellschaft*,
   27(3), 2985–2993.

3. Koshland, D. E. (1958). Application of a theory of enzyme
   specificity to protein synthesis. *Proceedings of the National
   Academy of Sciences*, 44(2), 98–104.

4. Connolly, M. L. (1983). Analytical molecular surface
   calculation. *Science*, 221(4612), 709–713.

5. Connolly, M. L. (1986). Shape complementarity at the hemoglobin
   α₁β₁ subunit interface. *Biopolymers*, 25, 1229–1247.

6. Lawrence, M. C. & Colman, P. M. (1993). Shape complementarity
   at protein/protein interfaces. *Journal of Molecular Biology*,
   234, 946–950.

7. do Carmo, M. P. (1976). *Differential Geometry of Curves and
   Surfaces*. Prentice-Hall.

8. Meyer, M., Desbrun, M., Schröder, P. & Barr, A. H. (2003).
   Discrete differential-geometry operators for triangulated
   2-manifolds. *Visualization and Mathematics III*, 35–57.

9. Bondi, A. (1964). van der Waals Volumes and Radii. *Journal of
   Physical Chemistry*, 68(3), 441–451.

10. Kyte, J. & Doolittle, R. F. (1982). A simple method for
    displaying the hydropathic character of a protein. *Journal of
    Molecular Biology*, 157(1), 105–132.

11. Gauss, C. F. (1828). *Disquisitiones generales circa
    superficies curvas*. Commentationes Societatis Regiae
    Scientiarum Gottingensis Recentiores, 6, 99–146.

12. Chern, S.-S. (1944). A simple intrinsic proof of the
    Gauss–Bonnet formula for closed Riemannian manifolds.
    *Annals of Mathematics*, 45(4), 747–752.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    page = render_sidebar()
    dispatch = {
        "🏠 Home": page_home,
        "🔷 The Shape Map": page_shape_map,
        "🗺️ The Topological Map": page_topological_map,
        "🔓 The Complementarity Test": page_complementarity,
        "📊 Surface Comparison": page_surface_comparison,
        "📚 Theory & Mathematics": page_theory,
    }
    handler = dispatch.get(page, page_home)
    handler()


if __name__ == "__main__":
    main()

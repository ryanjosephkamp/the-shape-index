"""
visualization.py — Dual Rendering Engine
==========================================

PlotlyRenderer  — interactive 3-D shape-indexed surfaces, patch maps,
                  complementarity overlays, curvedness heat maps.
MatplotlibRenderer — static publication-quality figures for the CLI
                     and scientific report.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.shape_engine import (
    TriMesh,
    CurvatureResult,
    ShapeIndexResult,
    PatchResult,
    SaddlePoint,
    ComplementarityResult,
    SHAPE_CATEGORIES,
    SHAPE_CATEGORY_COLORS,
)
from src.analysis import (
    FullShapeAnalysis,
    PatchStatistics,
    ComplementarityAnalysis,
    SaddlePointCatalogue,
    PresetComparisonResult,
)


# ═══════════════════════════════════════════════════════════════════════
# Colour helpers
# ═══════════════════════════════════════════════════════════════════════

# Blue (#0000FF, concave) → Grey (#CCCCCC, saddle) → Red (#FF0000, convex)
SHAPE_COLORSCALE = [
    [0.0,  "#0000FF"],
    [0.125, "#3366CC"],
    [0.25, "#6699CC"],
    [0.375, "#99CCCC"],
    [0.5,  "#CCCCCC"],
    [0.625, "#CCCC99"],
    [0.75, "#CC9966"],
    [0.875, "#CC6633"],
    [1.0,  "#FF0000"],
]

CURVEDNESS_COLORSCALE = "Viridis"

PATCH_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _si_to_color(si: float) -> str:
    """Map Shape Index [-1, +1] to a hex colour."""
    t = (si + 1.0) / 2.0  # normalise to [0, 1]
    t = max(0.0, min(1.0, t))
    # Interpolate blue → grey → red
    if t < 0.5:
        r = int(204 * (t / 0.5))
        g = int(204 * (t / 0.5))
        b = int(255 - (255 - 204) * (t / 0.5))
    else:
        s = (t - 0.5) / 0.5
        r = int(204 + (255 - 204) * s)
        g = int(204 * (1 - s))
        b = int(204 * (1 - s))
    return f"#{r:02x}{g:02x}{b:02x}"


# ═══════════════════════════════════════════════════════════════════════
# Plotly Renderer
# ═══════════════════════════════════════════════════════════════════════


class PlotlyRenderer:
    """Interactive Plotly visualizations."""

    # ── 1. Shape Index surface ────────────────────────────────────────
    @staticmethod
    def shape_index_surface(
        result: ShapeIndexResult,
        title: str = "Shape Index",
        show_edges: bool = False,
    ) -> go.Figure:
        """3-D mesh coloured by Shape Index."""
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces

        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=result.shape_index,
                colorscale=SHAPE_COLORSCALE,
                cmin=-1.0, cmax=1.0,
                colorbar=dict(title="Shape Index (S)", tickvals=[-1, -0.5, 0, 0.5, 1],
                              ticktext=["Cup −1", "Rut −0.5", "Saddle 0",
                                        "Ridge 0.5", "Cap +1"]),
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>"
                    "SI: %{intensity:.3f}<extra></extra>"
                ),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
        )
        return fig

    # ── 2. Curvedness surface ─────────────────────────────────────────
    @staticmethod
    def curvedness_surface(
        result: ShapeIndexResult,
        title: str = "Curvedness",
    ) -> go.Figure:
        """3-D mesh coloured by Curvedness."""
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces

        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=result.curvedness,
                colorscale=CURVEDNESS_COLORSCALE,
                colorbar=dict(title="Curvedness (C)"),
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>"
                    "C: %{intensity:.3f}<extra></extra>"
                ),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
        )
        return fig

    # ── 3. Patch map ──────────────────────────────────────────────────
    @staticmethod
    def patch_map(
        result: ShapeIndexResult,
        patches: PatchResult,
        title: str = "Surface Patches",
    ) -> go.Figure:
        """3-D mesh coloured by patch label."""
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces

        n_patches = patches.n_patches
        # Assign each vertex a pseudo-colour based on its patch label
        labels = patches.labels.astype(np.float64)

        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=labels,
                colorscale="Rainbow",
                colorbar=dict(title="Patch ID"),
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>"
                    "Patch: %{intensity:.0f}<extra></extra>"
                ),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
        )
        return fig

    # ── 4. Saddle point overlay ───────────────────────────────────────
    @staticmethod
    def saddle_overlay(
        result: ShapeIndexResult,
        saddle_points: List[SaddlePoint],
        title: str = "Saddle Points",
    ) -> go.Figure:
        """Shape Index surface with saddle points highlighted."""
        fig = PlotlyRenderer.shape_index_surface(result, title=title)

        if saddle_points:
            sp_pos = np.array([s.position for s in saddle_points])
            fig.add_trace(go.Scatter3d(
                x=sp_pos[:, 0], y=sp_pos[:, 1], z=sp_pos[:, 2],
                mode="markers",
                marker=dict(size=4, color="yellow", symbol="diamond",
                            line=dict(width=1, color="black")),
                name="Saddle Points",
                hovertemplate=(
                    "Saddle<br>x: %{x:.2f}<br>y: %{y:.2f}<br>"
                    "z: %{z:.2f}<extra></extra>"
                ),
            ))
        return fig

    # ── 5. Category histogram ─────────────────────────────────────────
    @staticmethod
    def category_histogram(
        analysis: FullShapeAnalysis,
        title: str = "Shape Category Distribution",
    ) -> go.Figure:
        """Bar chart of shape category fractions."""
        cats = list(analysis.category_fractions.keys())
        fracs = list(analysis.category_fractions.values())
        colors = [SHAPE_CATEGORY_COLORS.get(c, "#888") for c in cats]

        fig = go.Figure(data=[
            go.Bar(
                x=cats, y=fracs,
                marker_color=colors,
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        ])
        fig.update_layout(
            title=title,
            yaxis_title="Fraction of Vertices",
            xaxis_title="Shape Category",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=80),
        )
        return fig

    # ── 6. Shape Index histogram (continuous) ─────────────────────────
    @staticmethod
    def si_histogram(
        result: ShapeIndexResult,
        title: str = "Shape Index Distribution",
        n_bins: int = 50,
    ) -> go.Figure:
        """Histogram of Shape Index values."""
        fig = go.Figure(data=[
            go.Histogram(
                x=result.shape_index,
                nbinsx=n_bins,
                marker_color="#4A90D9",
                hovertemplate="S ∈ [%{x}]: %{y} vertices<extra></extra>",
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Shape Index (S)",
            yaxis_title="Count",
            template="plotly_white",
        )
        return fig

    # ── 7. Curvedness histogram ───────────────────────────────────────
    @staticmethod
    def curvedness_histogram(
        result: ShapeIndexResult,
        title: str = "Curvedness Distribution",
        n_bins: int = 50,
    ) -> go.Figure:
        """Histogram of Curvedness values."""
        fig = go.Figure(data=[
            go.Histogram(
                x=result.curvedness,
                nbinsx=n_bins,
                marker_color="#50C878",
                hovertemplate="C ∈ [%{x}]: %{y} vertices<extra></extra>",
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Curvedness (C)",
            yaxis_title="Count",
            template="plotly_white",
        )
        return fig

    # ── 8. Complementarity dual panel ─────────────────────────────────
    @staticmethod
    def complementarity_dual(
        comp: ComplementarityAnalysis,
        title: str = "Lock-and-Key Complementarity",
    ) -> go.Figure:
        """Side-by-side Shape Index surfaces for protein and ligand."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
            subplot_titles=["Protein (Cup / Concave)", "Ligand (Cap / Convex)"],
        )

        # Protein
        prot = comp.comp_result.protein_result
        pm = prot.mesh
        fig.add_trace(
            go.Mesh3d(
                x=pm.vertices[:, 0], y=pm.vertices[:, 1], z=pm.vertices[:, 2],
                i=pm.faces[:, 0], j=pm.faces[:, 1], k=pm.faces[:, 2],
                intensity=prot.shape_index,
                colorscale=SHAPE_COLORSCALE,
                cmin=-1.0, cmax=1.0,
                colorbar=dict(title="SI", x=0.45, len=0.6),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            ),
            row=1, col=1,
        )

        # Ligand
        lig = comp.comp_result.ligand_result
        lm = lig.mesh
        fig.add_trace(
            go.Mesh3d(
                x=lm.vertices[:, 0], y=lm.vertices[:, 1], z=lm.vertices[:, 2],
                i=lm.faces[:, 0], j=lm.faces[:, 1], k=lm.faces[:, 2],
                intensity=lig.shape_index,
                colorscale=SHAPE_COLORSCALE,
                cmin=-1.0, cmax=1.0,
                colorbar=dict(title="SI", x=1.0, len=0.6),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            margin=dict(l=0, r=0, t=60, b=0),
        )
        return fig

    # ── 9. Complementarity histogram overlay ──────────────────────────
    @staticmethod
    def complementarity_histograms(
        comp: ComplementarityAnalysis,
        title: str = "Shape Index Histograms — Protein vs Ligand",
    ) -> go.Figure:
        """Overlaid histograms of protein and ligand SI distributions."""
        cr = comp.comp_result
        bin_centres = (cr.histogram_bins[:-1] + cr.histogram_bins[1:]) / 2.0

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_centres, y=cr.protein_si_histogram,
            name="Protein", marker_color="rgba(0,0,255,0.5)",
            hovertemplate="SI=%{x:.2f}: %{y:.3f}<extra>Protein</extra>",
        ))
        fig.add_trace(go.Bar(
            x=bin_centres, y=cr.ligand_si_histogram,
            name="Ligand", marker_color="rgba(255,0,0,0.5)",
            hovertemplate="SI=%{x:.2f}: %{y:.3f}<extra>Ligand</extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Shape Index (S)",
            yaxis_title="Fraction",
            barmode="overlay",
            template="plotly_white",
        )
        return fig

    # ── 10. Preset comparison summary bar ─────────────────────────────
    @staticmethod
    def preset_comparison_bars(
        comparison: PresetComparisonResult,
        title: str = "Preset Surface Comparison",
    ) -> go.Figure:
        """Grouped bar chart: Mean SI and Mean C for each preset."""
        names = [r["Surface"] for r in comparison.summary_table]
        mean_si = [float(r["Mean SI"]) for r in comparison.summary_table]
        mean_c = [float(r["Mean C"]) for r in comparison.summary_table]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=mean_si, name="Mean Shape Index",
            marker_color="#4A90D9",
        ))
        fig.add_trace(go.Bar(
            x=names, y=mean_c, name="Mean Curvedness",
            marker_color="#50C878",
        ))
        fig.update_layout(
            title=title,
            barmode="group",
            yaxis_title="Value",
            template="plotly_white",
        )
        return fig

    # ── 11. Gaussian curvature surface ────────────────────────────────
    @staticmethod
    def gaussian_curvature_surface(
        result: ShapeIndexResult,
        title: str = "Gaussian Curvature",
    ) -> go.Figure:
        """3-D mesh coloured by Gaussian curvature K."""
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces
        K = result.curvature.gaussian_curvature

        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=K,
                colorscale="RdBu_r",
                colorbar=dict(title="K"),
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>"
                    "K: %{intensity:.4f}<extra></extra>"
                ),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
        )
        return fig

    # ── 12. Mean curvature surface ────────────────────────────────────
    @staticmethod
    def mean_curvature_surface(
        result: ShapeIndexResult,
        title: str = "Mean Curvature",
    ) -> go.Figure:
        """3-D mesh coloured by mean curvature H."""
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces
        H = result.curvature.mean_curvature

        fig = go.Figure(data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                intensity=H,
                colorscale="RdBu_r",
                colorbar=dict(title="H"),
                hovertemplate=(
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>"
                    "H: %{intensity:.4f}<extra></extra>"
                ),
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
        )
        return fig


# ═══════════════════════════════════════════════════════════════════════
# Matplotlib Renderer
# ═══════════════════════════════════════════════════════════════════════


class MatplotlibRenderer:
    """Static publication figures via Matplotlib."""

    # ── 1. Shape Index surface ────────────────────────────────────────
    @staticmethod
    def shape_index_surface(
        result: ShapeIndexResult,
        title: str = "Shape Index",
        figsize: Tuple[int, int] = (10, 8),
    ) -> Figure:
        """3-D triangulated surface coloured by Shape Index."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        mesh = result.mesh
        v = mesh.vertices
        f = mesh.faces

        # Build polygon collection
        cmap = plt.cm.coolwarm  # type: ignore
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        # Per-face colour: average of vertex SI
        face_si = result.shape_index[f].mean(axis=1)
        face_colors = cmap(norm(face_si))

        polys = v[f]
        pc = Poly3DCollection(polys, facecolors=face_colors,
                              edgecolors="none", alpha=0.9)
        ax.add_collection3d(pc)

        # Set limits
        _set_axis_limits(ax, v)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label="Shape Index (S)", shrink=0.6)
        fig.tight_layout()
        return fig

    # ── 2. Category bar chart ─────────────────────────────────────────
    @staticmethod
    def category_bar(
        analysis: FullShapeAnalysis,
        title: str = "Shape Category Distribution",
        figsize: Tuple[int, int] = (10, 5),
    ) -> Figure:
        """Horizontal bar chart of category fractions."""
        fig, ax = plt.subplots(figsize=figsize)
        cats = list(analysis.category_fractions.keys())
        fracs = list(analysis.category_fractions.values())
        colors = [SHAPE_CATEGORY_COLORS.get(c, "#888") for c in cats]

        ax.barh(cats, fracs, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Fraction of Vertices")
        ax.set_title(title, fontsize=13)
        ax.invert_yaxis()

        for i, (f_val, cat) in enumerate(zip(fracs, cats)):
            ax.text(f_val + 0.005, i, f"{f_val:.1%}", va="center", fontsize=9)

        fig.tight_layout()
        return fig

    # ── 3. SI histogram ───────────────────────────────────────────────
    @staticmethod
    def si_histogram(
        result: ShapeIndexResult,
        title: str = "Shape Index Distribution",
        figsize: Tuple[int, int] = (8, 5),
        n_bins: int = 50,
    ) -> Figure:
        """Histogram of Shape Index values."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(result.shape_index, bins=n_bins, color="#4A90D9",
                edgecolor="white", alpha=0.85)
        ax.set_xlabel("Shape Index (S)")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=13)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.6, label="Saddle (S=0)")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── 4. Complementarity histograms ─────────────────────────────────
    @staticmethod
    def complementarity_histograms(
        comp: ComplementarityAnalysis,
        title: str = "Protein vs Ligand SI",
        figsize: Tuple[int, int] = (8, 5),
    ) -> Figure:
        """Overlaid histograms for protein and ligand."""
        fig, ax = plt.subplots(figsize=figsize)
        cr = comp.comp_result
        bin_centres = (cr.histogram_bins[:-1] + cr.histogram_bins[1:]) / 2.0
        width = (cr.histogram_bins[1] - cr.histogram_bins[0]) * 0.8

        ax.bar(bin_centres, cr.protein_si_histogram, width=width,
               alpha=0.5, color="blue", label="Protein")
        ax.bar(bin_centres, cr.ligand_si_histogram, width=width,
               alpha=0.5, color="red", label="Ligand")
        ax.set_xlabel("Shape Index (S)")
        ax.set_ylabel("Fraction")
        ax.set_title(title, fontsize=13)
        ax.legend()
        fig.tight_layout()
        return fig

    # ── 5. Curvature scatter (κ₁ vs κ₂) ──────────────────────────────
    @staticmethod
    def curvature_scatter(
        result: ShapeIndexResult,
        title: str = "Principal Curvatures",
        figsize: Tuple[int, int] = (7, 7),
    ) -> Figure:
        """Scatter κ₁ vs κ₂ coloured by Shape Index."""
        fig, ax = plt.subplots(figsize=figsize)
        k1 = result.curvature.kappa1
        k2 = result.curvature.kappa2
        si = result.shape_index

        sc = ax.scatter(k2, k1, c=si, cmap="coolwarm", vmin=-1, vmax=1,
                        s=2, alpha=0.6)
        ax.set_xlabel("κ₂ (min)")
        ax.set_ylabel("κ₁ (max)")
        ax.set_title(title, fontsize=13)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.4)
        ax.plot([-10, 10], [-10, 10], "k--", alpha=0.3, label="κ₁=κ₂ (umbilical)")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="Shape Index (S)")
        fig.tight_layout()
        return fig

    # ── 6. Preset comparison table ────────────────────────────────────
    @staticmethod
    def preset_comparison(
        comparison: PresetComparisonResult,
        title: str = "Preset Comparison",
        figsize: Tuple[int, int] = (12, 5),
    ) -> Figure:
        """Grouped bar chart: Mean SI and Mean Curvedness."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        names = [r["Surface"] for r in comparison.summary_table]
        mean_si = [float(r["Mean SI"]) for r in comparison.summary_table]
        mean_c = [float(r["Mean C"]) for r in comparison.summary_table]

        x = np.arange(len(names))
        axes[0].bar(x, mean_si, color="#4A90D9")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean Shape Index")
        axes[0].set_title("Mean Shape Index by Surface")
        axes[0].axhline(0, color="grey", linestyle="--", alpha=0.5)

        axes[1].bar(x, mean_c, color="#50C878")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axes[1].set_ylabel("Mean Curvedness")
        axes[1].set_title("Mean Curvedness by Surface")

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        return fig


# ── Helper ────────────────────────────────────────────────────────────


def _set_axis_limits(ax, vertices: NDArray) -> None:
    """Set equal-aspect 3-D axis limits."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    mid = (vmin + vmax) / 2
    span = (vmax - vmin).max() / 2 * 1.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)  # type: ignore

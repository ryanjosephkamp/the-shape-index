"""
shape_engine.py — Core Differential Geometry Engine
=====================================================

Implements Koenderink's Shape Index, Curvedness, principal curvature
estimation on discrete triangle meshes, surface patch segmentation,
saddle-point detection, and protein–ligand complementarity analysis.

Key algorithms
--------------
- Vertex normal estimation (area-weighted face normals)
- Principal curvature estimation via the discrete Shape Operator
  (Hessian of the height function in the local tangent frame)
- Koenderink Shape Index:  S = (2/π) arctan((κ₁+κ₂)/(κ₁−κ₂))
- Curvedness:  C = √((κ₁² + κ₂²)/2)
- Patch segmentation by discretised Shape Index bins
- Saddle-point detection (|S| < threshold)
- Complementarity test (protein Cup ↔ ligand Cap)

References
----------
Koenderink, J. J. & van Doorn, A. J. (1992).
    Surface shape and curvature scales. *Image Vis. Comput.*, 10(8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ═══════════════════════════════════════════════════════════════════════
# Constants & defaults
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_PATCH_BINS: int = 9
"""Number of discrete Shape Index bins for patch segmentation."""

SHAPE_CATEGORIES: Dict[str, Tuple[float, float]] = {
    "Cup":             (-1.00, -0.75),
    "Trough":          (-0.75, -0.50),
    "Rut":             (-0.50, -0.25),
    "Saddle Rut":      (-0.25, -0.05),
    "Saddle":          (-0.05,  0.05),
    "Saddle Ridge":    ( 0.05,  0.25),
    "Ridge":           ( 0.25,  0.50),
    "Dome":            ( 0.50,  0.75),
    "Cap":             ( 0.75,  1.00),
}
"""
Koenderink's nine canonical shape categories.

Each entry maps a descriptive name to (S_low, S_high) bounds on the
Shape Index scale [-1, +1].
"""

SHAPE_CATEGORY_COLORS: Dict[str, str] = {
    "Cup":            "#0000FF",   # deep blue
    "Trough":         "#3366CC",   # medium blue
    "Rut":            "#6699CC",   # light blue
    "Saddle Rut":     "#99CCCC",   # cyan-ish
    "Saddle":         "#CCCCCC",   # grey
    "Saddle Ridge":   "#CCCC99",   # tan
    "Ridge":          "#CC9966",   # light orange
    "Dome":           "#CC6633",   # orange
    "Cap":            "#FF0000",   # deep red
}


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Vertex:
    """A single mesh vertex with position and optional properties."""
    x: float
    y: float
    z: float
    normal: Optional[NDArray] = None

    @property
    def position(self) -> NDArray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class TriMesh:
    """Triangle mesh representation.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    vertices : ndarray, shape (V, 3)
        Vertex positions.
    faces : ndarray, shape (F, 3)
        Triangle indices into *vertices*.
    """
    name: str
    vertices: NDArray   # (V, 3)
    faces: NDArray      # (F, 3), int

    # ── derived ──────────────────────────────────────────────────────
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def center_of_mass(self) -> NDArray:
        return self.vertices.mean(axis=0)

    def face_normals(self) -> NDArray:
        """Compute per-face normals (un-normalized)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return np.cross(v1 - v0, v2 - v0)

    def face_areas(self) -> NDArray:
        """Compute per-face areas."""
        fn = self.face_normals()
        return 0.5 * np.linalg.norm(fn, axis=1)

    def total_area(self) -> float:
        return float(self.face_areas().sum())

    def bounding_box(self) -> Tuple[NDArray, NDArray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@dataclass
class CurvatureResult:
    """Per-vertex curvature data.

    Fields
    ------
    kappa1 : ndarray, shape (V,)
        Maximum principal curvature at each vertex.
    kappa2 : ndarray, shape (V,)
        Minimum principal curvature at each vertex.
    mean_curvature : ndarray, shape (V,)
        H = (κ₁ + κ₂) / 2
    gaussian_curvature : ndarray, shape (V,)
        K = κ₁ · κ₂
    principal_dir1 : ndarray, shape (V, 3)
        Direction of κ₁.
    principal_dir2 : ndarray, shape (V, 3)
        Direction of κ₂.
    normals : ndarray, shape (V, 3)
        Estimated vertex normals.
    """
    kappa1: NDArray
    kappa2: NDArray
    mean_curvature: NDArray
    gaussian_curvature: NDArray
    principal_dir1: NDArray
    principal_dir2: NDArray
    normals: NDArray


@dataclass
class ShapeIndexResult:
    """Shape Index and Curvedness computed at every vertex.

    Fields
    ------
    shape_index : ndarray, shape (V,)
        Koenderink Shape Index ∈ [-1, +1].
    curvedness : ndarray, shape (V,)
        Curvedness C = √((κ₁² + κ₂²)/2).
    categories : list[str]
        Per-vertex shape category label.
    curvature : CurvatureResult
        Underlying curvature data.
    mesh : TriMesh
        The mesh over which the index was computed.
    """
    shape_index: NDArray
    curvedness: NDArray
    categories: List[str]
    curvature: CurvatureResult
    mesh: TriMesh


@dataclass
class PatchResult:
    """Result of surface patch segmentation.

    Fields
    ------
    labels : ndarray, shape (V,)
        Integer label per vertex identifying the connected patch.
    n_patches : int
        Total number of distinct patches.
    patch_areas : dict[int, float]
        Approximate area per patch (sum of 1/3 of adjacent face areas).
    patch_categories : dict[int, str]
        Dominant shape category of each patch.
    patch_centroids : dict[int, NDArray]
        Centroid (mean vertex position) of each patch.
    """
    labels: NDArray
    n_patches: int
    patch_areas: Dict[int, float]
    patch_categories: Dict[int, str]
    patch_centroids: Dict[int, NDArray]


@dataclass
class SaddlePoint:
    """A detected saddle point (or saddle region centroid).

    Fields
    ------
    vertex_index : int
        Index of the representative vertex.
    position : NDArray
        3-D coordinate.
    shape_index : float
        Shape Index value at this vertex.
    curvedness : float
        Curvedness at this vertex.
    patch_id : int
        Patch the vertex belongs to.
    """
    vertex_index: int
    position: NDArray
    shape_index: float
    curvedness: float
    patch_id: int


@dataclass
class ComplementarityResult:
    """Result of protein–ligand shape complementarity analysis.

    Fields
    ------
    protein_result : ShapeIndexResult
        Shape analysis of the protein surface.
    ligand_result : ShapeIndexResult
        Shape analysis of the ligand surface.
    protein_cup_fraction : float
        Fraction of protein vertices classified as Cup/Trough.
    ligand_cap_fraction : float
        Fraction of ligand vertices classified as Cap/Dome.
    complementarity_score : float
        Mean complementarity ∈ [0, 1]; 1 = perfect lock-and-key.
    protein_si_histogram : NDArray
        Histogram of protein Shape Index values.
    ligand_si_histogram : NDArray
        Histogram of ligand Shape Index values.
    histogram_bins : NDArray
        Bin edges for the histograms.
    explanation : str
        A human-readable explanation of the complementarity.
    """
    protein_result: ShapeIndexResult
    ligand_result: ShapeIndexResult
    protein_cup_fraction: float
    ligand_cap_fraction: float
    complementarity_score: float
    protein_si_histogram: NDArray
    ligand_si_histogram: NDArray
    histogram_bins: NDArray
    explanation: str


# ═══════════════════════════════════════════════════════════════════════
# Mesh construction helpers
# ═══════════════════════════════════════════════════════════════════════


def _triangulate_grid(n_u: int, n_v: int) -> NDArray:
    """Create triangle indices for a (n_u × n_v) grid of vertices.

    Returns shape (2·(n_u-1)·(n_v-1), 3) int array.
    """
    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            p00 = i * n_v + j
            p01 = i * n_v + (j + 1)
            p10 = (i + 1) * n_v + j
            p11 = (i + 1) * n_v + (j + 1)
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])
    return np.array(faces, dtype=np.int64)


def build_sphere(
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_u: int = 40,
    n_v: int = 40,
    name: str = "Sphere",
) -> TriMesh:
    """Build a UV-sphere mesh."""
    u = np.linspace(0, np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = radius * np.sin(U) * np.cos(V) + center[0]
    Y = radius * np.sin(U) * np.sin(V) + center[1]
    Z = radius * np.cos(U) + center[2]
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n_u, n_v)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_ellipsoid(
    a: float = 2.0,
    b: float = 1.0,
    c: float = 0.5,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_u: int = 40,
    n_v: int = 40,
    name: str = "Ellipsoid",
) -> TriMesh:
    """Build an ellipsoidal mesh with semi-axes (a, b, c)."""
    u = np.linspace(0, np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = a * np.sin(U) * np.cos(V) + center[0]
    Y = b * np.sin(U) * np.sin(V) + center[1]
    Z = c * np.cos(U) + center[2]
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n_u, n_v)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_saddle(
    scale: float = 2.0,
    n: int = 40,
    name: str = "Saddle (Hyperbolic Paraboloid)",
) -> TriMesh:
    """Build a hyperbolic paraboloid z = x² − y²."""
    t = np.linspace(-scale, scale, n)
    X, Y = np.meshgrid(t, t, indexing="ij")
    Z = X ** 2 - Y ** 2
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n, n)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_torus(
    R: float = 2.0,
    r: float = 0.6,
    n_u: int = 50,
    n_v: int = 30,
    name: str = "Torus",
) -> TriMesh:
    """Build a torus with major radius R and tube radius r."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n_u, n_v)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_double_sphere(
    radius: float = 1.0,
    separation: float = 1.5,
    n_u: int = 30,
    n_v: int = 30,
    name: str = "Double Sphere (Protein–Ligand Mock)",
) -> Tuple[TriMesh, TriMesh]:
    """Build two spheres that simulate a protein pocket (concave)
    and ligand (convex), separated by *separation*.

    Returns (protein_mesh, ligand_mesh).
    The protein is a sphere with a concave dent (Cap → Cup when
    viewed from the binding interface), and the ligand is a smaller
    convex sphere that fits into it.
    """
    # Protein: larger sphere with a concave dent
    protein = build_sphere(
        radius=radius * 1.5,
        center=(-separation / 2, 0.0, 0.0),
        n_u=n_u, n_v=n_v,
        name="Protein Surface",
    )
    # Deform protein: push vertices near +x inward to create a cup
    dx = protein.vertices[:, 0] - (-separation / 2)
    dy = protein.vertices[:, 1]
    dz = protein.vertices[:, 2]
    dist_from_pole = np.sqrt(dy**2 + dz**2)
    # Concave dent toward +x face
    mask = (dx > radius * 0.6) & (dist_from_pole < radius * 0.9)
    dent_depth = 0.4 * radius * np.exp(-(dist_from_pole[mask]**2) / (0.5 * radius)**2)
    protein.vertices[mask, 0] -= dent_depth

    # Ligand: smaller convex sphere
    ligand = build_sphere(
        radius=radius * 0.6,
        center=(separation / 2, 0.0, 0.0),
        n_u=n_u, n_v=n_v,
        name="Ligand Surface",
    )
    return protein, ligand


def build_wavy_surface(
    amplitude: float = 0.5,
    frequency: float = 2.0,
    size: float = 3.0,
    n: int = 50,
    name: str = "Wavy Surface",
) -> TriMesh:
    """Build a wavy height-field z = A·sin(f·x)·cos(f·y).

    Good for testing mixed curvature regions.
    """
    t = np.linspace(-size, size, n)
    X, Y = np.meshgrid(t, t, indexing="ij")
    Z = amplitude * np.sin(frequency * X) * np.cos(frequency * Y)
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n, n)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_binding_pocket(
    pocket_depth: float = 1.0,
    pocket_radius: float = 1.5,
    surface_radius: float = 3.0,
    n: int = 50,
    name: str = "Binding Pocket",
) -> TriMesh:
    """Build a flat surface with a Gaussian concave pocket in the center."""
    t = np.linspace(-surface_radius, surface_radius, n)
    X, Y = np.meshgrid(t, t, indexing="ij")
    r2 = X**2 + Y**2
    Z = -pocket_depth * np.exp(-r2 / pocket_radius**2)
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n, n)
    return TriMesh(name=name, vertices=vertices, faces=faces)


def build_bump(
    bump_height: float = 1.0,
    bump_radius: float = 1.5,
    surface_radius: float = 3.0,
    n: int = 50,
    name: str = "Convex Bump (Ligand Mock)",
) -> TriMesh:
    """Build a flat surface with a Gaussian convex bump — the complement
    of a binding pocket."""
    t = np.linspace(-surface_radius, surface_radius, n)
    X, Y = np.meshgrid(t, t, indexing="ij")
    r2 = X**2 + Y**2
    Z = bump_height * np.exp(-r2 / bump_radius**2)
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = _triangulate_grid(n, n)
    return TriMesh(name=name, vertices=vertices, faces=faces)


# ═══════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════


def _vertex_adjacency(mesh: TriMesh) -> Dict[int, set]:
    """Build vertex → set-of-neighbor-vertices adjacency map."""
    adj: Dict[int, set] = {i: set() for i in range(mesh.n_vertices)}
    for f in mesh.faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _vertex_faces(mesh: TriMesh) -> Dict[int, List[int]]:
    """Build vertex → list-of-face-indices map."""
    vf: Dict[int, List[int]] = {i: [] for i in range(mesh.n_vertices)}
    for fi, f in enumerate(mesh.faces):
        for vi in f:
            vf[vi].append(fi)
    return vf


# ═══════════════════════════════════════════════════════════════════════
# Normal estimation
# ═══════════════════════════════════════════════════════════════════════


def estimate_vertex_normals(mesh: TriMesh) -> NDArray:
    """Compute area-weighted vertex normals.

    Returns shape (V, 3) array of unit normals.
    """
    fn = mesh.face_normals()                       # (F, 3)
    normals = np.zeros_like(mesh.vertices)         # (V, 3)
    for fi, f in enumerate(mesh.faces):
        for vi in f:
            normals[vi] += fn[fi]                  # area-weighted
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return normals / norms


# ═══════════════════════════════════════════════════════════════════════
# Principal curvature estimation
# ═══════════════════════════════════════════════════════════════════════


def estimate_principal_curvatures(mesh: TriMesh) -> CurvatureResult:
    """Estimate principal curvatures at every vertex.

    Uses the discrete shape-operator approach:
    1. For each vertex, compute a local tangent frame (e1, e2, n).
    2. Fit a quadratic patch to the 1-ring neighbourhood.
    3. Extract the shape operator (Hessian of the height function)
       and diagonalise to get (κ₁, κ₂).

    Returns a CurvatureResult dataclass.
    """
    V = mesh.n_vertices
    normals = estimate_vertex_normals(mesh)
    adj = _vertex_adjacency(mesh)

    kappa1 = np.zeros(V)
    kappa2 = np.zeros(V)
    pdir1 = np.zeros((V, 3))
    pdir2 = np.zeros((V, 3))

    for vi in range(V):
        n = normals[vi]
        p = mesh.vertices[vi]
        neighbours = list(adj[vi])
        if len(neighbours) < 3:
            # Not enough neighbours for a fit
            continue

        # Local tangent frame
        # Pick an arbitrary tangent vector
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, arbitrary)
        e1 /= np.linalg.norm(e1) + 1e-15
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2) + 1e-15

        # Project neighbours into local frame
        diffs = mesh.vertices[neighbours] - p  # (K, 3)
        u_vals = diffs @ e1                    # local x
        v_vals = diffs @ e2                    # local y
        h_vals = diffs @ n                     # height

        # Fit quadratic: h ≈ a·u² + b·u·v + c·v²
        # Build design matrix [u², u·v, v²]
        A_mat = np.column_stack([
            u_vals**2,
            u_vals * v_vals,
            v_vals**2,
        ])

        # Least-squares solve
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, h_vals, rcond=None)
        except np.linalg.LinAlgError:
            continue

        a, b, c = coeffs
        # Shape operator (Weingarten map) in (e1, e2) basis:
        # S = -[[2a, b], [b, 2c]]
        # (negative sign: height function h decreases for convex)
        S = np.array([[-2 * a, -b],
                      [-b, -2 * c]])

        # Eigenvalues = principal curvatures
        eigvals, eigvecs = np.linalg.eigh(S)
        # Convention: κ₁ ≥ κ₂
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        kappa1[vi] = eigvals[0]
        kappa2[vi] = eigvals[1]
        pdir1[vi] = eigvecs[0, 0] * e1 + eigvecs[1, 0] * e2
        pdir2[vi] = eigvecs[0, 1] * e1 + eigvecs[1, 1] * e2

    mean_curv = (kappa1 + kappa2) / 2.0
    gauss_curv = kappa1 * kappa2

    return CurvatureResult(
        kappa1=kappa1,
        kappa2=kappa2,
        mean_curvature=mean_curv,
        gaussian_curvature=gauss_curv,
        principal_dir1=pdir1,
        principal_dir2=pdir2,
        normals=normals,
    )


# ═══════════════════════════════════════════════════════════════════════
# Shape Index & Curvedness
# ═══════════════════════════════════════════════════════════════════════


def compute_shape_index(curvature: CurvatureResult) -> NDArray:
    """Compute Koenderink's Shape Index at every vertex.

    S = (2/π) · arctan((κ₁ + κ₂) / (κ₁ − κ₂))

    Returns ndarray of shape (V,) with values in [-1, +1].
    Flat points (κ₁ ≈ κ₂ ≈ 0) are assigned S = 0.
    """
    k1 = curvature.kappa1
    k2 = curvature.kappa2
    denom = k1 - k2
    numer = k1 + k2

    # Avoid division by zero at umbilical points (κ₁ ≈ κ₂)
    flat = np.abs(denom) < 1e-12
    safe_denom = np.where(flat, 1.0, denom)
    S = (2.0 / np.pi) * np.arctan(numer / safe_denom)
    S = np.where(flat, 0.0, S)
    return np.clip(S, -1.0, 1.0)


def compute_curvedness(curvature: CurvatureResult) -> NDArray:
    """Compute Curvedness at every vertex.

    C = √((κ₁² + κ₂²) / 2)

    Returns ndarray of shape (V,) with non-negative values.
    """
    return np.sqrt((curvature.kappa1 ** 2 + curvature.kappa2 ** 2) / 2.0)


def classify_shape(shape_index: float) -> str:
    """Classify a single Shape Index value into one of nine categories."""
    for name, (lo, hi) in SHAPE_CATEGORIES.items():
        if lo <= shape_index < hi:
            return name
    # Edge cases
    if shape_index >= 1.0 - 1e-9:
        return "Cap"
    if shape_index <= -1.0 + 1e-9:
        return "Cup"
    return "Saddle"


def classify_shape_array(shape_index: NDArray) -> List[str]:
    """Classify an array of Shape Index values."""
    return [classify_shape(float(s)) for s in shape_index]


# ═══════════════════════════════════════════════════════════════════════
# Full Shape Index computation pipeline
# ═══════════════════════════════════════════════════════════════════════


def compute_full_shape_index(mesh: TriMesh) -> ShapeIndexResult:
    """Run the full pipeline: curvature → Shape Index → categories.

    Parameters
    ----------
    mesh : TriMesh
        Input triangle mesh.

    Returns
    -------
    ShapeIndexResult
    """
    curv = estimate_principal_curvatures(mesh)
    si = compute_shape_index(curv)
    C = compute_curvedness(curv)
    cats = classify_shape_array(si)
    return ShapeIndexResult(
        shape_index=si,
        curvedness=C,
        categories=cats,
        curvature=curv,
        mesh=mesh,
    )


# ═══════════════════════════════════════════════════════════════════════
# Patch segmentation
# ═══════════════════════════════════════════════════════════════════════


def segment_patches(
    mesh: TriMesh,
    shape_index: NDArray,
    n_bins: int = DEFAULT_PATCH_BINS,
) -> PatchResult:
    """Segment the surface into connected patches of similar shape.

    Algorithm:
    1. Bin each vertex by its Shape Index into *n_bins* equal bands.
    2. Within each bin, find connected components (BFS on the mesh
       adjacency restricted to the bin).
    3. Compute area & centroid per patch.

    Returns a PatchResult.
    """
    V = mesh.n_vertices
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    vertex_bin = np.digitize(shape_index, bin_edges) - 1
    vertex_bin = np.clip(vertex_bin, 0, n_bins - 1)

    adj = _vertex_adjacency(mesh)
    labels = -np.ones(V, dtype=np.int64)
    current_label = 0

    for b in range(n_bins):
        in_bin = set(np.where(vertex_bin == b)[0])
        visited = set()
        for seed in in_bin:
            if seed in visited:
                continue
            # BFS
            queue = [seed]
            component = []
            while queue:
                v = queue.pop(0)
                if v in visited:
                    continue
                visited.add(v)
                component.append(v)
                for nb in adj[v]:
                    if nb in in_bin and nb not in visited:
                        queue.append(nb)
            if component:
                for v in component:
                    labels[v] = current_label
                current_label += 1

    # Unlabelled vertices (shouldn't happen, but just in case)
    for vi in range(V):
        if labels[vi] < 0:
            labels[vi] = current_label
            current_label += 1

    n_patches = current_label

    # Per-patch area estimation (vertex area = sum of 1/3 of adjacent face areas)
    face_area = mesh.face_areas()
    vertex_area = np.zeros(V)
    for fi, f in enumerate(mesh.faces):
        for vi in f:
            vertex_area[vi] += face_area[fi] / 3.0

    patch_areas: Dict[int, float] = {}
    patch_centroids: Dict[int, NDArray] = {}
    patch_categories: Dict[int, str] = {}

    for pid in range(n_patches):
        mask = labels == pid
        if not np.any(mask):
            continue
        patch_areas[pid] = float(vertex_area[mask].sum())
        patch_centroids[pid] = mesh.vertices[mask].mean(axis=0)
        # Dominant category: mode of shape categories
        si_vals = shape_index[mask]
        mean_si = float(si_vals.mean())
        patch_categories[pid] = classify_shape(mean_si)

    return PatchResult(
        labels=labels,
        n_patches=n_patches,
        patch_areas=patch_areas,
        patch_categories=patch_categories,
        patch_centroids=patch_centroids,
    )


# ═══════════════════════════════════════════════════════════════════════
# Saddle-point detection
# ═══════════════════════════════════════════════════════════════════════


def detect_saddle_points(
    result: ShapeIndexResult,
    threshold: float = 0.10,
    patches: Optional[PatchResult] = None,
) -> List[SaddlePoint]:
    """Detect vertices with near-zero Shape Index (saddle regions).

    Parameters
    ----------
    result : ShapeIndexResult
        Full shape analysis result.
    threshold : float
        |S| < threshold identifies saddle vertices.
    patches : PatchResult, optional
        If provided, each saddle point records its patch_id.

    Returns
    -------
    list[SaddlePoint]
    """
    si = result.shape_index
    C = result.curvedness
    mesh = result.mesh

    saddle_mask = np.abs(si) < threshold
    indices = np.where(saddle_mask)[0]

    points: List[SaddlePoint] = []
    for vi in indices:
        pid = int(patches.labels[vi]) if patches is not None else -1
        points.append(SaddlePoint(
            vertex_index=int(vi),
            position=mesh.vertices[vi].copy(),
            shape_index=float(si[vi]),
            curvedness=float(C[vi]),
            patch_id=pid,
        ))
    return points


# ═══════════════════════════════════════════════════════════════════════
# Complementarity analysis
# ═══════════════════════════════════════════════════════════════════════


def complementarity_analysis(
    protein_mesh: TriMesh,
    ligand_mesh: TriMesh,
    n_bins: int = 50,
) -> ComplementarityResult:
    """Analyse shape complementarity between protein and ligand surfaces.

    The Lock-and-Key test: the protein binding site should be
    predominantly concave (Cup / Trough, S < -0.25) and the ligand
    contact surface should be predominantly convex (Cap / Dome, S > 0.25).

    The complementarity score is defined as:
        score = (protein_cup_fraction + ligand_cap_fraction) / 2

    Parameters
    ----------
    protein_mesh, ligand_mesh : TriMesh
        The two surfaces to compare.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    ComplementarityResult
    """
    prot_result = compute_full_shape_index(protein_mesh)
    lig_result = compute_full_shape_index(ligand_mesh)

    # Protein: fraction of vertices with S < -0.25 (concave)
    prot_concave = np.mean(prot_result.shape_index < -0.25)
    # Ligand: fraction of vertices with S > 0.25 (convex)
    lig_convex = np.mean(lig_result.shape_index > 0.25)

    score = (prot_concave + lig_convex) / 2.0

    # Histograms
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    prot_hist, _ = np.histogram(prot_result.shape_index, bins=bin_edges)
    lig_hist, _ = np.histogram(lig_result.shape_index, bins=bin_edges)

    # Normalise to fractions
    prot_hist = prot_hist.astype(np.float64) / max(prot_hist.sum(), 1)
    lig_hist = lig_hist.astype(np.float64) / max(lig_hist.sum(), 1)

    explanation = (
        f"Protein concave fraction (S < -0.25): {prot_concave:.1%}\n"
        f"Ligand convex fraction  (S >  0.25): {lig_convex:.1%}\n"
        f"Complementarity score: {score:.3f}\n\n"
    )
    if score > 0.6:
        explanation += (
            "Strong shape complementarity consistent with "
            "lock-and-key binding."
        )
    elif score > 0.4:
        explanation += (
            "Moderate complementarity — induced-fit may play a role."
        )
    else:
        explanation += (
            "Low complementarity — surfaces are not shape-matched."
        )

    return ComplementarityResult(
        protein_result=prot_result,
        ligand_result=lig_result,
        protein_cup_fraction=float(prot_concave),
        ligand_cap_fraction=float(lig_convex),
        complementarity_score=float(score),
        protein_si_histogram=prot_hist,
        ligand_si_histogram=lig_hist,
        histogram_bins=bin_edges,
        explanation=explanation,
    )


# ═══════════════════════════════════════════════════════════════════════
# Preset surfaces
# ═══════════════════════════════════════════════════════════════════════


def get_preset_surfaces() -> Dict[str, TriMesh]:
    """Return a dict of named preset surfaces for demonstration."""
    return {
        "Sphere": build_sphere(radius=1.0, name="Sphere"),
        "Ellipsoid": build_ellipsoid(a=2.0, b=1.0, c=0.5, name="Ellipsoid"),
        "Saddle": build_saddle(scale=2.0, name="Saddle (Hyperbolic Paraboloid)"),
        "Torus": build_torus(R=2.0, r=0.6, name="Torus"),
        "Wavy Surface": build_wavy_surface(name="Wavy Surface"),
        "Binding Pocket": build_binding_pocket(name="Binding Pocket"),
    }

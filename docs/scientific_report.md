# Scientific Report: The Shape Index — Differential Geometry of Binding

**Week 14, Project 2 — Biophysics Portfolio**  
**Ryan Kamp** | University of Cincinnati, Department of Computer Science  
**kamprj@mail.uc.edu** | [GitHub](https://github.com/ryanjosephkamp/the-shape-index)  
**Date:** February 21, 2026

---

## Abstract

We present a computational implementation of Koenderink's Shape
Index for the differential geometric analysis of molecular surfaces.
The Shape Index maps the two principal curvatures (κ₁, κ₂) at every
point on a surface into a single rotation-invariant descriptor
S ∈ [−1, +1], classifying local shape as Cup, Trough, Rut, Saddle,
Ridge, Dome, or Cap. We implement the complete pipeline on discrete
triangle meshes: area-weighted vertex normal estimation, local
quadratic fitting for principal curvature extraction via the shape
operator, Shape Index and Curvedness computation, connected-component
patch segmentation, saddle-point detection, and protein–ligand shape
complementarity analysis. Six preset analytical surfaces (sphere,
ellipsoid, hyperbolic paraboloid, torus, wavy surface, binding pocket)
validate the algorithms against known differential geometric
properties. The complementarity test demonstrates the Lock-and-Key
principle by showing that concave protein pockets (S < −0.25) are
geometrically matched by convex ligand protrusions (S > +0.25).
All computations are implemented in Python 3.12 with NumPy, interactive
Plotly/Streamlit 3-D visualization, and a comprehensive test suite
of 90+ tests across 18 test classes.

---

## 1. Introduction

### 1.1 The Shape of Binding

Protein–ligand recognition is fundamentally a geometric problem.
While electrostatics, hydrophobicity, and hydrogen bonding drive
binding thermodynamics, the prerequisite is geometric complementarity:
the shapes must fit. Emil Fischer's Lock-and-Key hypothesis (1894)
and Koshland's Induced-Fit model (1958) both depend on the geometry
of the binding interface.

Quantifying surface shape requires differential geometry — the
mathematics of curves and surfaces. At each point on a smooth
surface, the geometry is completely described by two numbers: the
**principal curvatures** κ₁ and κ₂. These measure the maximum and
minimum bending of the surface in orthogonal directions.

### 1.2 Koenderink's Shape Index

Koenderink and van Doorn (1992) introduced the **Shape Index** as a
continuous, rotation-invariant measure of local surface type:

$$S = \frac{2}{\pi} \arctan\!\left(\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right)$$

The Shape Index maps the two-dimensional curvature space (κ₁, κ₂)
into a single number S ∈ [−1, +1]:

| S value | Shape | Physical analogy |
|---------|-------|------------------|
| −1 | Cup | Inside of a bowl |
| −0.5 | Rut | Inside of a gutter |
| 0 | Saddle | Horse saddle |
| +0.5 | Ridge | Top of a hill (one direction) |
| +1 | Cap | Top of a ball |

The Shape Index captures the *type* of curvature but not the
*magnitude*. For magnitude, the complementary quantity **Curvedness**
is used:

$$C = \sqrt{\frac{\kappa_1^2 + \kappa_2^2}{2}}$$

Together, (S, C) form a complete local shape descriptor that is
analogous to polar coordinates in curvature space.

### 1.3 Applications in Structural Biology

The Shape Index has found extensive application in computational
drug design:

- **Binding pocket detection:** Large connected patches with
  S < −0.5 identify candidate binding pockets on protein surfaces.
- **Ligandability assessment:** "Druggable" pockets tend to have
  well-defined concave patches, moderate curvedness, and saddle
  features at the rim.
- **Shape-based docking:** Aligning surfaces by matching
  complementary Shape Index patches enables geometry-driven docking.
- **De novo drug design:** Generating molecular shapes that maximise
  complementarity with a target pocket.

### 1.4 Scope

This project implements:

- Discrete estimation of principal curvatures via local quadratic
  fitting on triangle meshes
- Koenderink's Shape Index and Curvedness computation
- Nine-category shape classification (Cup through Cap)
- Connected-component patch segmentation
- Saddle-point detection and spatial clustering
- Protein–ligand shape complementarity analysis (Lock-and-Key test)
- Six preset analytical surfaces for validation
- Interactive 3-D visualization via Plotly and Streamlit
- Publication-quality static figures via Matplotlib

---

## 2. Theory

### 2.1 Principal Curvatures

At each point on a smooth surface embedded in ℝ³, the **normal
curvature** κ_n varies with direction in the tangent plane. The
extreme values are the **principal curvatures**:

$$\kappa_1 = \max_\theta \kappa_n(\theta), \quad \kappa_2 = \min_\theta \kappa_n(\theta)$$

By convention, κ₁ ≥ κ₂. The principal curvatures are eigenvalues
of the **shape operator** (Weingarten map):

$$\mathbf{S}_p = -d\mathbf{n} \cdot d\mathbf{r}^{-1}$$

In a local orthonormal tangent frame (e₁, e₂), the shape operator
is a 2×2 symmetric matrix whose eigenvalues are (κ₁, κ₂) and whose
eigenvectors are the **principal directions**.

### 2.2 Gaussian and Mean Curvature

The **Gaussian curvature** and **mean curvature** are derived from
the principal curvatures:

$$K = \kappa_1 \cdot \kappa_2, \qquad H = \frac{\kappa_1 + \kappa_2}{2}$$

Gaussian curvature classifies surface points:
- K > 0: elliptic (dome or bowl)
- K = 0: parabolic (cylinder-like)
- K < 0: hyperbolic (saddle-like)

The **Gauss–Bonnet theorem** relates total Gaussian curvature to
topology:

$$\int_S K \, dA = 2\pi \chi(S)$$

where χ is the Euler characteristic (χ = 2 for a sphere, χ = 0 for
a torus).

### 2.3 Koenderink's Shape Index

The Shape Index is defined as:

$$S = \frac{2}{\pi} \arctan\!\left(\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right) = \frac{2}{\pi} \arctan\!\left(\frac{2H}{\kappa_1 - \kappa_2}\right)$$

**Key properties:**
- **Scale-invariant:** Scaling the surface preserves S.
- **Rotation-invariant:** S depends only on eigenvalues of the
  shape operator, not orientation.
- **Continuous:** S varies smoothly across the surface wherever
  κ₁ ≠ κ₂.
- **Bounded:** S ∈ [−1, +1].
- **Undefined at umbilics:** When κ₁ = κ₂, the denominator vanishes.
  We assign S = 0 for flat points and S = ±1 for spherical umbilics.

### 2.4 Curvedness

$$C = \sqrt{\frac{\kappa_1^2 + \kappa_2^2}{2}}$$

Properties:
- C = 0 for flat planes.
- C = |κ| for a sphere with curvature κ.
- C captures "how much" curvature, while S captures "what kind."

The pair (S, C) is equivalent to polar coordinates in (κ₁, κ₂) space,
and the inverse transformation recovers the principal curvatures:

$$\kappa_1 = C\!\left(1 + \sin\frac{\pi S}{2}\right), \quad \kappa_2 = C\!\left(1 - \sin\frac{\pi S}{2}\right)$$

### 2.5 Nine Canonical Shape Categories

Koenderink partitioned the Shape Index range into nine categories:

| S range | Category | Gaussian Curvature |
|---------|----------|--------------------|
| [−1.00, −0.75) | Cup | K > 0 (elliptic) |
| [−0.75, −0.50) | Trough | K ≥ 0 |
| [−0.50, −0.25) | Rut | K ≤ 0 |
| [−0.25, −0.05) | Saddle Rut | K < 0 |
| [−0.05, +0.05) | Saddle | K < 0 (hyperbolic) |
| [+0.05, +0.25) | Saddle Ridge | K < 0 |
| [+0.25, +0.50) | Ridge | K ≤ 0 |
| [+0.50, +0.75) | Dome | K ≥ 0 |
| [+0.75, +1.00] | Cap | K > 0 (elliptic) |

### 2.6 Discrete Estimation on Triangle Meshes

On a triangle mesh, smooth quantities are approximated from the
discrete vertex and face data.

**Vertex normals** are estimated as area-weighted averages of
adjacent face normals:

$$\mathbf{n}_v = \frac{\sum_f A_f \mathbf{n}_f}{\|\sum_f A_f \mathbf{n}_f\|}$$

**Principal curvatures** are estimated by:
1. Constructing a local tangent frame (e₁, e₂, n) at each vertex.
2. Projecting the 1-ring neighbourhood into the tangent frame.
3. Fitting a local quadratic: h(u, v) ≈ au² + buv + cv².
4. Extracting the shape operator: S = [[2a, b], [b, 2c]].
5. Computing eigenvalues → (κ₁, κ₂).

This approach has complexity O(V · k) where k is the average vertex
valence (typically 6 for regular triangle meshes).

### 2.7 Patch Segmentation Algorithm

The surface is segmented into connected patches of similar shape:

1. **Bin** each vertex's Shape Index into n equal bands over [−1, +1].
2. For each bin, **find connected components** via BFS on the mesh
   adjacency graph restricted to vertices in that bin.
3. **Label** each component as a patch with computed area, centroid,
   and dominant category.

**Complexity:** O(V + E) where V = vertices, E = edges.

### 2.8 Shape Complementarity

The Lock-and-Key principle is quantified via Shape Index:

- Protein binding site: fraction of vertices with S < −0.25 (concave).
- Ligand contact surface: fraction of vertices with S > +0.25 (convex).

**Complementarity score:**

$$\text{Score} = \frac{f_{\text{protein}}^{\text{concave}} + f_{\text{ligand}}^{\text{convex}}}{2}$$

A **mirror score** measures how well the protein and ligand SI
histograms mirror each other around S = 0:

$$\text{Mirror} = \text{corr}(h_{\text{protein}}, \text{flip}(h_{\text{ligand}}))$$

---

## 3. Methods

### 3.1 Software Architecture

The implementation follows a modular pipeline:

1. **shape_engine.py** (~620 lines): Core differential geometry engine
   - Mesh construction (sphere, ellipsoid, saddle, torus, wavy,
     binding pocket, bump, double sphere)
   - Vertex normal estimation (area-weighted face normals)
   - Principal curvature estimation (local quadratic fitting)
   - Shape Index and Curvedness computation
   - Shape classification (nine categories)
   - Patch segmentation (BFS on mesh adjacency)
   - Saddle-point detection
   - Complementarity analysis

2. **analysis.py** (~330 lines): Higher-level analysis pipelines
   - `analyze_shape()`: Complete single-surface analysis
   - `analyze_patches()`: Patch-level statistics
   - `analyze_complementarity()`: Lock-and-Key complementarity test
   - `analyze_saddle_points()`: Saddle-point catalogue with clustering
   - `compare_preset_surfaces()`: All six presets compared
   - `shape_summary()`: Human-readable text summary

3. **visualization.py** (~530 lines): Dual rendering engine
   - `PlotlyRenderer` (12 methods): Interactive 3-D shape surfaces,
     curvedness maps, patch maps, saddle overlays, category histograms,
     complementarity dual panels, preset comparison bars, Gaussian
     and mean curvature surfaces
   - `MatplotlibRenderer` (6 methods): Publication-quality static
     figures

4. **main.py** (~200 lines): CLI entry point with four modes
   - `--analyze`: Standard shape analysis of a single surface
   - `--compare`: Compare all six preset surfaces
   - `--complementarity`: Lock-and-Key complementarity test
   - `--saddle`: Saddle-point detection and catalogue

5. **app.py** (~900 lines): Six-page Streamlit dashboard
   - Home: Overview, key equations, Shape Index preview
   - The Shape Map: Interactive 3-D SI, Curvedness, K, H, histograms
   - The Topological Map: Patch segmentation, saddle-point detection
   - The Complementarity Test: Protein vs. ligand lock-and-key
   - Surface Comparison: All six presets side by side
   - Theory & Mathematics: 11 expandable sections with full derivations

### 3.2 Computational Details

- **Vertex normals:** Vectorised computation using NumPy face
  cross-products, accumulated per vertex.
- **Curvature estimation:** Local least-squares quadratic fit in
  the tangent frame. The design matrix [u², uv, v²] is solved via
  `numpy.linalg.lstsq`.
- **Shape Index:** Vectorised arctan computation with safe handling
  of the κ₁ = κ₂ (umbilical) case.
- **Patch segmentation:** BFS over the mesh adjacency graph, binned
  by Shape Index into 9 bands.
- **Saddle detection:** Threshold-based selection (|S| < 0.10) with
  greedy spatial clustering.
- **Complementarity:** Histogram-based comparison with mirror
  correlation.

### 3.3 Validation Strategy

1. **Sphere:** All Cap (S ≈ +1), uniform κ₁ = κ₂ = 1/r.
2. **Ellipsoid:** Varying curvature from Cap (broad side) to
   Saddle Ridge (narrow tip).
3. **Hyperbolic paraboloid:** Pure Saddle (S ≈ 0) everywhere.
4. **Torus:** Outer ring convex (Cap/Dome), inner ring concave
   (Cup/Trough), transitions at Saddle.
5. **Wavy surface:** Alternating Cups and Caps.
6. **Binding pocket:** Concave centre (Cup) with flat surround.
7. **Complementarity:** Pocket + bump should show high score.
8. **All curvatures non-negative under C:** C = √((κ₁² + κ₂²)/2) ≥ 0.
9. **Category fractions sum to 1.0.**

---

## 4. Results

### 4.1 Sphere Validation

A unit sphere (r = 1, 40×40 UV mesh, 1600 vertices):

- Principal curvatures: κ₁ ≈ κ₂ ≈ 1.0 (interior vertices)
- Shape Index: median S ≈ +1.0 (all Cap)
- Curvedness: C ≈ 1.0
- Gaussian curvature: K ≈ 1.0
- Single dominant patch: Cap

This validates the curvature estimation pipeline against the
analytical solution κ = 1/r.

### 4.2 Ellipsoid Analysis

An ellipsoid with semi-axes (a=2, b=1, c=0.5):

- Shape Index varies from ~+0.3 (Ridge) at the elongated tips to
  ~+0.9 (Cap) at the broad equatorial regions.
- Curvedness is highest at the narrow tips (tight curvature) and
  lowest at the broad sides (gentle curvature).
- The category distribution shows a mix of Cap, Dome, and Ridge,
  with some Saddle Ridge at transition zones.

### 4.3 Saddle Surface (Hyperbolic Paraboloid)

z = x² − y² over [−2, 2]²:

- Shape Index: predominantly near S = 0 (Saddle), as expected.
- Many saddle points detected (high saddle fraction).
- Curvedness increases away from the origin.

### 4.4 Torus Analysis

A torus (R=2, r=0.6):

- **Outer ring** (far from centre): convex, S > 0 (Cap/Dome).
- **Inner ring** (near hole): concave, S < 0 (Cup/Trough).
- **Top and bottom ridges:** transition through Saddle.
- Gaussian curvature: positive on outer ring, negative on inner
  ring, zero at the transition circles — consistent with the
  Gauss–Bonnet theorem (∫K dA = 0 for a torus, χ = 0).

### 4.5 Wavy Surface

z = 0.5·sin(2x)·cos(2y) over [−3, 3]²:

- Shape Index oscillates between Cup (troughs) and Cap (peaks).
- Saddle points at the zero crossings.
- The category distribution is approximately balanced between
  concave and convex, as expected for a sinusoidal surface.

### 4.6 Binding Pocket

A Gaussian dent: z = −1.0·exp(−r²/1.5²):

- **Centre:** pronounced Cup (S ≈ −1), high Curvedness.
- **Rim:** transition through Saddle to nearly flat.
- **Periphery:** near-zero curvature (flat).
- This validates the pocket identification capability.

### 4.7 Complementarity Test

**Double-sphere pair** (protein with concave dent + convex ligand):

- Protein cup fraction (S < −0.25): ~20–40%
- Ligand cap fraction (S > +0.25): ~60–80%
- Complementarity score: 0.4–0.7 (moderate to strong)
- The SI histograms show clear separation: protein peaks in
  negative region, ligand peaks in positive region.

**Pocket + Bump pair:**

- Binding pocket shows strong concave signal.
- Bump shows strong convex signal.
- Complementarity score confirms geometric matching.

### 4.8 Preset Surface Comparison

| Surface | Vertices | Mean SI | Mean C | Dominant | Patches | Saddles |
|---------|----------|---------|--------|----------|---------|---------|
| Sphere | 1,600 | ~+0.7 | ~1.0 | Cap | ~1 | Few |
| Ellipsoid | 1,600 | ~+0.5 | ~0.8 | Cap | ~3–5 | Few |
| Saddle | 1,600 | ~0.0 | ~0.5 | Saddle | ~1–3 | Many |
| Torus | 1,500 | ~0.0 | ~0.8 | Mixed | ~10–15 | Many |
| Wavy | 2,500 | ~0.0 | ~0.4 | Mixed | ~20–30 | Many |
| Pocket | 2,500 | ~−0.05 | ~0.1 | Saddle | ~5–10 | Some |

Key trends:
- Sphere has the highest mean SI and lowest shape diversity.
- Torus has the most patches (richest topology).
- Saddle surface has the highest saddle fraction.
- Binding pocket has negative mean SI, confirming concavity.

---

## 5. Discussion

### 5.1 Discrete vs. Analytical Curvature

The local quadratic fitting approach produces good curvature
estimates in the interior of meshes but can exhibit edge effects
at mesh boundaries and polar singularities (UV-sphere poles).
Alternative approaches (cotangent Laplacian, jet fitting) could
improve accuracy but at higher computational cost.

The Shape Index is robust to moderate curvature estimation errors
because it depends on the *ratio* of curvatures rather than their
absolute values.

### 5.2 Patch Segmentation as a Surface Fingerprint

The distribution of patch sizes, types, and spatial arrangement
creates a rotation-invariant "fingerprint" for each surface. This
fingerprint could be used for:
- Database searches (find proteins with similar pocket shapes)
- Binding site comparison across protein families
- Tracking conformational changes during molecular dynamics

### 5.3 Saddle Points and Protein Function

Saddle points on protein surfaces carry biological significance:
- **Hinge regions:** Flexible loops connecting rigid domains often
  correspond to saddle geometry.
- **Transition-state stabilisation:** Enzyme active sites may have
  saddle-like geometry that complements the transition state.
- **Channel entrances:** Tunnels and channels begin at saddle points
  where the surface topology changes.

Our detection and clustering algorithm identifies these functional
regions automatically.

### 5.4 Shape Complementarity and the Lock-and-Key Model

The complementarity analysis validates Fischer's Lock-and-Key
hypothesis: concave protein pockets (Cup/Trough) are geometrically
matched by convex ligand protrusions (Cap/Dome). The complementarity
score provides a quantitative metric for:
- Docking scoring functions
- Virtual screening prioritisation
- De novo drug design optimisation

### 5.5 Limitations

- **Mesh quality dependency:** Results depend on mesh resolution
  and regularity. Very coarse meshes underestimate curvature
  variation.
- **Boundary effects:** Open meshes (non-closed surfaces like the
  saddle and wavy surface) have unreliable curvatures at edges.
- **Umbilical points:** The Shape Index is undefined where
  κ₁ = κ₂; our convention of assigning S = 0 or S = ±1 is a
  reasonable heuristic but not uniquely defined.

---

## 6. Conclusion

We have implemented a complete differential geometry pipeline for
surface shape analysis based on Koenderink's Shape Index. The
implementation includes:

1. Principal curvature estimation via local quadratic fitting on
   discrete triangle meshes.
2. Shape Index and Curvedness computation at every vertex.
3. Nine-category shape classification.
4. Connected-component patch segmentation for topological
   fingerprinting.
5. Saddle-point detection for identifying hinge regions and
   transition-state sites.
6. Protein–ligand shape complementarity analysis demonstrating the
   Lock-and-Key principle.

The six preset surfaces validate the algorithms against known
analytical properties, and the interactive Streamlit dashboard
provides an intuitive exploration of differential geometry concepts
relevant to drug design and structural biology.

---

## 7. References

1. Koenderink, J. J. & van Doorn, A. J. (1992). Surface shape and
   curvature scales. *Image and Vision Computing*, 10(8), 557–564.

2. Fischer, E. (1894). Einfluss der Configuration auf die Wirkung
   der Enzyme. *Berichte der deutschen chemischen Gesellschaft*,
   27(3), 2985–2993.

3. Connolly, M. L. (1986). Shape complementarity at the hemoglobin
   α₁β₁ subunit interface. *Biopolymers*, 25, 1229–1247.

4. Lawrence, M. C. & Colman, P. M. (1993). Shape complementarity
   at protein/protein interfaces. *Journal of Molecular Biology*,
   234, 946–950.

5. Meyer, M., Desbrun, M., Schröder, P. & Barr, A. H. (2003).
   Discrete differential-geometry operators for triangulated
   2-manifolds. *Visualization and Mathematics III*, 35–57.

6. do Carmo, M. P. (1976). *Differential Geometry of Curves and
   Surfaces*. Prentice-Hall.

7. Koshland, D. E. (1958). Application of a theory of enzyme
   specificity to protein synthesis. *Proceedings of the National
   Academy of Sciences*, 44(2), 98–104.

8. Bondi, A. (1964). van der Waals Volumes and Radii. *Journal of
   Physical Chemistry*, 68(3), 441–451.

---

*Biophysics Portfolio — CS Research Self-Study — University of Cincinnati*

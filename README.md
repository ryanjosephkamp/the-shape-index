# 🔷 The Shape Index — Differential Geometry of Binding

**Week 14, Project 2** | Biophysics Portfolio  
**Ryan Kamp** | University of Cincinnati, Department of Computer Science  
kamprj@mail.uc.edu | [GitHub](https://github.com/ryanjosephkamp/the-shape-index)

---

## Overview

Map two principal curvatures to one rotation-invariant number,
paint the surface blue-to-red, and watch binding pockets
light up — that's the Shape Index.

This project implements Koenderink's Shape Index for differential
geometric analysis of surfaces, covering:

- **Principal Curvatures** — local quadratic fitting on triangle meshes
- **Shape Index** — continuous Cup-to-Cap mapping S ∈ [−1, +1]
- **Curvedness** — magnitude of curvature C ≥ 0
- **Nine-Category Classification** — Cup, Trough, Rut, Saddle Rut, Saddle, Saddle Ridge, Ridge, Dome, Cap
- **Patch Segmentation** — BFS-based connected-component decomposition
- **Saddle-Point Detection** — identifying hinge regions with spatial clustering
- **Lock-and-Key Complementarity** — protein pocket vs. ligand bump shape matching
- **Six preset surfaces** — sphere, ellipsoid, saddle, torus, wavy surface, binding pocket

---

## Quick Start

```bash
# Navigate to the project directory
cd week_14_projects/week_14_project_2

# Activate the virtual environment
source ../../.venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run the default analysis
python main.py

# Run the Streamlit dashboard
streamlit run app.py

# Run the test suite
pytest tests/ -v
```

---

## Project Structure

```
week_14_project_2/
├── src/
│   ├── __init__.py             # Package facade (re-exports all symbols)
│   ├── shape_engine.py         # Core differential geometry engine (~620 lines)
│   ├── analysis.py             # Analysis pipelines (~330 lines)
│   └── visualization.py        # Plotly + Matplotlib rendering (~530 lines)
├── tests/
│   └── test_shape_index.py     # 18 classes, 90+ tests
├── docs/
│   ├── scientific_report.md    # Full scientific report
│   └── w14p2_shape_index_ieee.tex  # IEEE conference paper
├── figures/                    # Generated figures (auto-created)
├── main.py                     # CLI entry point (4 modes)
├── app.py                      # Streamlit dashboard (6 pages, ~900 lines)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## The Science

### Koenderink's Shape Index

The Shape Index maps two principal curvatures into a single
rotation-invariant descriptor:

$$S = \frac{2}{\pi} \arctan\!\left(\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right)$$

### Key Equations

| Quantity | Equation |
|----------|----------|
| Shape Index | S = (2/π) arctan((κ₁+κ₂)/(κ₁−κ₂)) |
| Curvedness | C = √((κ₁² + κ₂²)/2) |
| Gaussian curvature | K = κ₁ · κ₂ |
| Mean curvature | H = (κ₁ + κ₂)/2 |
| Complementarity | Score = (f_concave + f_convex)/2 |
| Gauss–Bonnet | ∫K dA = 2πχ |

### Nine Shape Categories

| S range | Category | Shape |
|---------|----------|-------|
| [−1.00, −0.75) | Cup | Inside of a bowl |
| [−0.75, −0.50) | Trough | Inside of a gutter |
| [−0.50, −0.25) | Rut | Groove |
| [−0.25, −0.05) | Saddle Rut | Near-saddle concave |
| [−0.05, +0.05) | Saddle | Horse saddle |
| [+0.05, +0.25) | Saddle Ridge | Near-saddle convex |
| [+0.25, +0.50) | Ridge | Top of a hill (1 dir) |
| [+0.50, +0.75) | Dome | Gentle hill |
| [+0.75, +1.00] | Cap | Top of a ball |

### Six Preset Surfaces

| Surface | Description |
|---------|-------------|
| Sphere | Unit sphere, uniform κ = 1 |
| Ellipsoid | Semi-axes (2, 1, 0.5), varying curvature |
| Saddle | Hyperbolic paraboloid z = x² − y² |
| Torus | R = 2, r = 0.6, concave inner + convex outer |
| Wavy Surface | z = 0.5 sin(2x) cos(2y) |
| Binding Pocket | Gaussian dent z = −exp(−r²/1.5²) |

---

## CLI Usage

```bash
# Default: analyze sphere surface
python main.py

# Analyze a specific surface
python main.py --analyze --surface torus --save --verbose

# Compare all six preset surfaces
python main.py --compare --save

# Lock-and-Key complementarity test
python main.py --complementarity --save

# Saddle-point detection
python main.py --saddle --surface torus --save

# Verbose output
python main.py --analyze --verbose
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--analyze` | Standard shape analysis | ✓ |
| `--compare` | Compare all 6 presets | |
| `--complementarity` | Lock-and-Key test | |
| `--saddle` | Saddle-point detection | |
| `--surface NAME` | Preset surface | sphere |
| `--save` | Save figures to `figures/` | |
| `--verbose` | Verbose output | |

---

## Streamlit Dashboard

```bash
streamlit run app.py
```

### Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview, key equations, Shape Index preview, science dropdowns |
| 🗺️ **The Shape Map** | Interactive 3-D Shape Index surface, Curvedness, Gaussian/Mean curvature, histograms, informational dropdowns |
| 🧩 **The Topological Map** | Patch segmentation visualization, saddle-point detection, category distribution, informational dropdowns |
| 🔑 **The Complementarity Test** | Side-by-side protein vs. ligand panels, histogram comparison, complementarity score, informational dropdowns |
| 📊 **Surface Comparison** | All 6 presets compared, bar charts, summary tables, informational dropdowns |
| 📚 **Theory & Mathematics** | 11 expandable sections: principal curvatures, Shape Index, Curvedness, Gaussian/Mean curvature, patch segmentation, saddle detection, complementarity, shape operator, discrete differential geometry, drug design, mathematical relations |

Every visualization and metric panel includes an **ℹ️ informational dropdown** explaining what you're seeing, how to interpret the data, and why it matters — 20+ dropdowns across 40+ total expanders.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_shape_index.py::TestShapeIndex -v
```

### Test Coverage

- **18 test classes**, **90+ test methods** covering all modules
- Engine tests: Constants, Vertex, TriMesh, MeshBuilders (12 tests),
  Normals, Curvature (5 tests), ShapeIndex (5 tests),
  Classification (10 tests), FullPipeline, PatchSegmentation,
  SaddleDetection, Complementarity
- Analysis tests: AnalyzeShape, PatchStatistics,
  ComplementarityAnalysis, SaddleCatalogue, PresetComparison,
  ShapeSummary
- Visualization tests: PlotlyRenderer (12 methods),
  MatplotlibRenderer (6 methods)
- CLI tests: argument parsing for all modes and flags

---

## Dependencies

- **Python ≥ 3.10**
- **NumPy** — numerical computation
- **SciPy** — scientific computing
- **Matplotlib** — static publication figures
- **Plotly** — interactive HTML visualization
- **Streamlit** — web dashboard
- **Pandas** — data tables
- **pytest** — testing framework

---

## References

1. Koenderink, J. J. & van Doorn, A. J. (1992). *Image Vision Comput.*, 10(8), 557–564.
2. Fischer, E. (1894). *Ber. Dtsch. Chem. Ges.*, 27(3), 2985–2993.
3. Connolly, M. L. (1986). *Biopolymers*, 25, 1229–1247.
4. Lawrence, M. C. & Colman, P. M. (1993). *J. Mol. Biol.*, 234, 946–950.
5. Meyer, M. et al. (2003). *Vis. Math. III*, 35–57.
6. do Carmo, M. P. (1976). *Differential Geometry of Curves and Surfaces*. Prentice-Hall.

---

*Biophysics Portfolio — CS Research Self-Study — University of Cincinnati*

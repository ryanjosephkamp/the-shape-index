# Week 14 - Project 2: "The Shape Index" – Differential Geometry of Binding

## Overview

**Week:** 14 (Apr 21 – Apr 28)  
**Theme:** Computational Geometry, Differential Geometry, and Graph Theory  
**Goal:** Implement the algorithms that turn "Atomic Coordinates" into "Learnable Features" (Curvature, SASA, Geodesics).

---

## Project Details

### The "Gap" It Fills
Mastery of **Differential Geometry** and **Topological Fingerprinting**.

Binding pockets aren't just "holes." They have specific shapes: Rut, Ridge, Saddle, Cap, Cup.

You will implement **Koenderink's Shape Index**, a mathematical transformation that maps the Principal Curvatures (κ₁, κ₂) into a single "Shape" value from -1 to +1.

### The Concept
- **Input:** The mesh from Week 7.
- **The Math:**
  - Calculate Principal Curvatures at every vertex.
  - Compute **Shape Index (S)**: S = (2/π) × arctan((κ₁+κ₂)/(κ₁-κ₂)).
  - Compute **Curvedness (C)**: Total magnitude of curvature.
- **The Output:** A "fingerprint" of the protein surface that is invariant to rotation.

### Novelty/Creative Angle
**"The Topological Map":**
- Don't just color it. Discretize the surface into "Patches."
- Identify "Saddle Points" specifically. In enzymology, saddle points on the surface often indicate dynamic hinge regions or transition state stabilization sites.

### Technical Implementation
- **Language:** Python (Trimesh or PyVista).
- **Math:** Estimating the Hessian Matrix on a discrete mesh.

### The "Paper" & Interactive Element
- *Interactive:* "The Complementarity Test." Load a Protein-Ligand complex. "Open" the book (separate them). Show that the Protein's surface is "Blue/Cup" and the Ligand's surface is "Red/Cap." This visually proves the **Lock-and-Key theory**.
- *Paper Focus:* "Topological Descriptors for Ligandability: Surface Segmentation using Koenderink's Shape Index."

---

## Progress Tracking

- [ ] Initial research and planning
- [ ] Core implementation
- [ ] Testing and validation
- [ ] Documentation and paper draft
- [ ] Interactive demo creation

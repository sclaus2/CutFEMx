CutFEMx Documentation
=====================

.. raw:: html

   <div class="hero-section">
     <div class="hero-content">
       <div class="hero-logo">
         <img src="_static/cutfemx_logo_small.png" alt="CutFEMx Logo" class="hero-logo-img">
       </div>
       <h1 class="hero-title">CutFEMx</h1>
       <p class="hero-subtitle">Cut Finite Element Methods for FEniCSx</p>
       <p class="hero-subtitle">Advanced computational methods for complex geometries and embedded boundary problems</p>
       <div class="hero-buttons">
         <a href="getting-started/installation.html" class="btn-hero btn-hero-primary">
           🚀 Get Started
         </a>
         <a href="examples/index.html" class="btn-hero btn-hero-secondary">
           💡 Examples
         </a>
         <a href="https://github.com/sclaus2/CutFEMx" class="btn-hero btn-hero-secondary">
           📱 GitHub
         </a>
       </div>
     </div>
   </div>

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <span class="feature-icon">🔲</span>
       <h3 class="feature-title">Implicit Geometries</h3>
       <p class="feature-description">Define complex shapes using level set functions without conforming mesh generation</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">⚖️</span>
       <h3 class="feature-title">Stabilization Methods</h3>
       <p class="feature-description">Employ ghost penalty or element aggregation to ensure stable computations</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">🌊</span>
       <h3 class="feature-title">Moving Boundaries</h3>
       <p class="feature-description">Efficiently handle time-dependent geometries with embedded boundary methods</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">🔗</span>
       <h3 class="feature-title">FEniCSx Integration</h3>
       <p class="feature-description">Seamless integration with the modern FEniCSx finite element framework</p>
     </div>
   </div>

Overview
--------

CutFEMx is a library that extends FEniCSx with cut finite element methods, enabling the solution of partial differential equations on complex domains without the need for conforming mesh generation. This approach is particularly valuable for:

• **Complex Geometries**: Solve problems on intricate shapes defined implicitly through level set functions
• **Moving Boundaries**: Handle evolving domains in fluid-structure interaction and multi-physics problems  
• **Embedded Methods**: Implement fictitious domain and immersed boundary techniques
• **Multi-Material Problems**: Handle interfaces and material boundaries naturally

The library provides a high-level interface that makes cut finite element methods accessible while maintaining the flexibility and performance of the underlying FEniCSx framework.


Key Features
------------

Cut Mesh Generation
~~~~~~~~~~~~~~~~~~~

CutFEMx automatically identifies and handles different element types:

- **Cut elements**: Elements intersected by the boundary
- **Inside elements**: Elements fully inside the domain  
- **Outside elements**: Elements fully outside the domain

The library provides efficient algorithms for:

- Boundary reconstruction from level set functions
- Sub-cell integration using adaptive quadrature
- Interface tracking and normal computation

Stabilization Techniques
~~~~~~~~~~~~~~~~~~~~~~~~

To ensure well-posed problems, CutFEMx includes several stabilization methods:

**Ghost Penalty Method**
  Adds penalty terms on element faces to control small cuts

**Aggregate Cells**
  Combines small cut elements into larger aggregates to improve stability

Integration with FEniCSx Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CutFEMx is designed to work seamlessly with:

- **DOLFINx**: Core finite element library
- **UFL**: Unified Form Language for weak forms
- **Basix**: Finite element basis functions
- **PETSc**: Linear algebra and solvers
- **PyVista**: Visualization and post-processing

Applications
------------

CutFEMx is particularly well-suited for:

**Fluid-Structure Interaction**
  Model complex moving boundaries without remeshing

**Topology Optimization**  
  Handle evolving material distributions efficiently

**Multi-Phase Flow**
  Track interfaces in fluid dynamics problems

**Biomedical Simulations**
  Model complex anatomical geometries from medical imaging


Getting Started
---------------

.. raw:: html

   <div class="grid-container">
   <div class="grid-item">
   <h3>1. Installation</h3>
   <p>Set up CutFEMx in your environment</p>
   <a href="getting-started/installation.html" class="btn-primary">Install CutFEMx</a>
   </div>

   <div class="grid-item">
   <h3>2. First Steps</h3>
   <p>Build your first cut FEM program</p>
   <a href="getting-started/quickstart.html" class="btn-primary">Quickstart Tutorial</a>
   </div>

   </div>


Citation
--------

If you use CutFEMx in your research, please cite:

.. code-block:: bibtex

   @article{CutFEM2015,
     title={CutFEM: discretizing geometry and partial differential equations},
     author={Burman, Erik and Claus, Susanne and Hansbo, Peter and 
             Larson, Mats G and Massing, Andr{\'e}},
     journal={International Journal for Numerical Methods in Engineering},
     volume={104},
     number={7},
     pages={472--501},
     year={2015},
     publisher={Wiley Online Library}
   }

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Installation Guide

   getting-started/installation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Quickstart

   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user-guide/index
   user-guide/level-sets
   user-guide/element-classification
   user-guide/cut-meshes
   user-guide/stabilization
   user-guide/boundary-conditions
   user-guide/dof-constraints
   user-guide/quadrature

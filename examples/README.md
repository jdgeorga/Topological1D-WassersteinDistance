Examples

This directory contains example scripts and notebooks demonstrating how to use the Wasserstein distance analysis code.

Directory Structure
------------------
examples/
├── notebooks/              # Interactive tutorials
│   ├── 1_model_corruption.ipynb
│   ├── 2_structure_relaxation.ipynb
│   ├── 3_distance_calculation.ipynb
│   └── 4_visualization.ipynb
└── scripts/               # Standalone example scripts
    ├── corrupt_single_model.py
    ├── relax_single_structure.py
    ├── calculate_single_distance.py
    └── create_custom_plot.py

Scripts
-------
1. Model Corruption (corrupt_single_model.py)
Shows how to corrupt a single ML model:

from src.corruption.generate import ModelCorruptor

corruptor = ModelCorruptor(
    base_model_path="models/lmp_sw_mos2.pth",
    output_dir="corrupted_models/example"
)
corruptor.corrupt_model(seed=42)

2. Structure Relaxation (relax_single_structure.py)
Shows how to relax a structure using corrupted models:

from src.relaxation.allegro import StructureOptimizer

optimizer = StructureOptimizer(
    intralayer_models={
        'layer1': "models/lmp_sw_mos2.pth",
        'layer2': "models/lmp_sw_wse2.pth"
    },
    interlayer_model="models/lmp_kc_wse2_mos2.pth",
    layer_symbols=[["Mo", "S", "S"], ["W", "Se", "Se"]]
)
relaxed_atoms, final_energy = optimizer.relax_structure(atoms, "example_relaxation")

3. Distance Calculation (calculate_single_distance.py)
Shows how to calculate Wasserstein distance:

from src.metrics.voronoi import VoronoiAnalyzer
from src.metrics.wasserstein import WassersteinMetric

voronoi = VoronoiAnalyzer()
wasserstein = WassersteinMetric(lattice_vectors)
distance = wasserstein.calculate_distance(points1, points2)

4. Visualization (create_custom_plot.py)
Shows how to create publication-quality plots:

from src.visualization.plots import DistancePlotter

plotter = DistancePlotter()
plotter.plot_distance_vs_corruption(
    distances,
    "example_plot.pdf",
    title="Example Plot"
)

Notebooks
---------
The notebooks provide detailed, step-by-step tutorials:

1. 1_model_corruption.ipynb: Interactive tutorial on model corruption
2. 2_structure_relaxation.ipynb: Guide to structure relaxation with corrupted models
3. 3_distance_calculation.ipynb: Tutorial on Wasserstein distance calculations
4. 4_visualization.ipynb: Examples of creating various plots and visualizations

Running the Examples
------------------
1. Make sure you have installed the package:
   pip install -e .

2. Run a script:
   python examples/scripts/corrupt_single_model.py

3. Or launch Jupyter to run the notebooks:
   jupyter notebook examples/notebooks/

Required Data
------------
The examples use the following input files:

1. Structure Files:
   - data/input/structures/MoS2_WSe2_1D.xyz
   - data/input/structures/MoS2_WSe2_2D.xyz

2. Model Files:
   - data/input/potentials/lmp_sw_mos2.pth
   - data/input/potentials/lmp_sw_wse2.pth
   - data/input/potentials/lmp_kc_wse2_mos2.pth
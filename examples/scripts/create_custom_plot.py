#!/usr/bin/env python
"""
Example script showing how to create custom plots.
"""
from pathlib import Path
import numpy as np
from src.visualization.plots import DistancePlotter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create some example data
    corruption_factors = np.logspace(-4.3, -0.3, 12)
    distances = np.random.rand(12) * 0.1
    
    # Initialize plotter
    plotter = DistancePlotter()
    
    # Create distance vs corruption plot
    plotter.plot_distance_vs_corruption(
        distances,
        output_path="example_plot.pdf",
        title="Example Distance Plot",
        gradient_thresholds={"threshold": 1e-2}
    )
    
    # Create comparison plot
    distances2 = distances * 1.5 + np.random.rand(12) * 0.05
    plotter.plot_distance_comparison(
        distances,
        distances2,
        output_path="example_comparison.pdf",
        labels=("Dataset 1", "Dataset 2"),
        title="Example Comparison Plot"
    )
    
    logger.info("Plots saved to example_plot.pdf and example_comparison.pdf")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Example script showing how to calculate Wasserstein distance.
"""
from pathlib import Path
import numpy as np
from ase.io import read
from src.metrics.voronoi import VoronoiAnalyzer
from src.metrics.wasserstein import WassersteinMetric
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load structures
    structure = read("relaxed_structures/example_relaxation.xyz")
    reference = read("structures/MoS2_WSe2_1D.xyz")
    
    # Set up analyzers
    lattice_vectors = np.array([[3.1841, 0.0], [-1.5920, 2.7575]])
    voronoi = VoronoiAnalyzer()
    wasserstein = WassersteinMetric(lattice_vectors)
    
    # Calculate distances
    pristine_cell = voronoi.get_primitive_voronoi_cell(lattice_vectors)
    
    # Get points
    structure_points = structure.positions[structure.arrays['atom_types'] == 0, :2]
    reference_points = reference.positions[reference.arrays['atom_types'] == 0, :2]
    query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]
    
    # Calculate interpolated points
    interpolated_points = voronoi.interpolate_displacements(
        structure_points,
        reference_points,
        query_points,
        pristine_cell
    )
    
    # Calculate distance
    distance, _, _, _ = wasserstein.calculate_distance(
        interpolated_points[0],
        interpolated_points[1]
    )
    
    logger.info(f"Wasserstein distance: {distance:.6f} Å")

if __name__ == "__main__":
    main()

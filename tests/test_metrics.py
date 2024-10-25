import pytest
import numpy as np
from src.metrics.voronoi import VoronoiAnalyzer
from src.metrics.wasserstein import WassersteinMetric

def test_voronoi_cell_generation(mock_lattice_vectors):
    """Test primitive Voronoi cell generation."""
    analyzer = VoronoiAnalyzer()
    cell = analyzer.get_primitive_voronoi_cell(mock_lattice_vectors)
    
    assert cell is not None
    assert isinstance(cell, np.ndarray)
    assert cell.shape[1] == 2

def test_wasserstein_metric(mock_lattice_vectors):
    """Test Wasserstein distance calculation."""
    metric = WassersteinMetric(mock_lattice_vectors)
    
    # Create test displacements
    disp1 = np.random.randn(10, 2)
    disp2 = np.random.randn(10, 2)
    
    distance, transport_matrix, _, _ = metric.calculate_distance(disp1, disp2)
    
    assert isinstance(distance, float)
    assert distance >= 0
    assert transport_matrix.shape == (10, 10)

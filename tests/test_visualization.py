import pytest
import numpy as np
from src.visualization.plots import DistancePlotter

def test_plotter_initialization(test_data_dir):
    """Test DistancePlotter initialization."""
    plotter = DistancePlotter()
    assert np.allclose(plotter.corruption_factors, np.logspace(-4.3, -0.3, 12))

def test_distance_plot_generation(test_data_dir):
    """Test distance vs corruption plot generation."""
    plotter = DistancePlotter()
    distances = np.random.rand(12)
    output_path = test_data_dir / "test_plot.pdf"
    
    plotter.plot_distance_vs_corruption(
        distances,
        str(output_path),
        title="Test Plot"
    )
    
    assert output_path.exists()

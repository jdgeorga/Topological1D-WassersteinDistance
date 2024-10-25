import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_lattice_vectors():
    """Generate mock lattice vectors."""
    return np.array([[3.1841, 0.0], [-1.5920, 2.7575]])

@pytest.fixture
def mock_model():
    """Create a mock neural network model."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)
            
    model = MockModel()
    return model

@pytest.fixture
def mock_structure():
    """Create a mock atomic structure."""
    from ase import Atoms
    import numpy as np
    
    # Create a simple MoS2-WSe2 structure
    positions = np.array([
        [0.0, 0.0, 12.91],  # Mo
        [1.0, 0.0, 14.47],  # S
        [1.0, 0.0, 11.35],  # S
        [0.0, 0.0, 18.86],  # W
        [1.0, 0.0, 20.54],  # Se
        [1.0, 0.0, 17.18],  # Se
    ])
    
    symbols = ['Mo', 'S', 'S', 'W', 'Se', 'Se']
    cell = [[3.1841, 0.0, 0.0],
            [-1.5920, 2.7575, 0.0],
            [0.0, 0.0, 31.8857]]
    
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=[True, True, False])
    atoms.arrays['atom_types'] = np.array([0, 1, 2, 3, 4, 5])
    
    return atoms

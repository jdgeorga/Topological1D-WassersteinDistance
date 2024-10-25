import pytest
from src.relaxation.allegro import StructureOptimizer

def test_structure_optimizer_initialization(mock_structure, test_data_dir):
    """Test StructureOptimizer initialization."""
    intralayer_models = {
        'layer1': str(test_data_dir / "mock_mos2.pth"),
        'layer2': str(test_data_dir / "mock_wse2.pth")
    }
    interlayer_model = str(test_data_dir / "mock_interlayer.pth")
    
    optimizer = StructureOptimizer(
        intralayer_models=intralayer_models,
        interlayer_model=interlayer_model,
        layer_symbols=[["Mo", "S", "S"], ["W", "Se", "Se"]],
        device="cpu"
    )
    
    assert optimizer.layer_symbols == [["Mo", "S", "S"], ["W", "Se", "Se"]]
    assert optimizer.device == "cpu"

def test_calculator_setup(mock_structure):
    """Test calculator setup."""
    with pytest.raises(ValueError):
        # Should raise error for missing atom_types
        atoms = mock_structure.copy()
        del atoms.arrays['atom_types']
        optimizer = StructureOptimizer(...)
        optimizer.setup_calculators(atoms)

import pytest
import torch
import numpy as np
from src.corruption.generate import ModelCorruptor

def test_model_corruptor_initialization(test_data_dir, mock_model):
    """Test ModelCorruptor initialization."""
    # Save mock model
    model_path = test_data_dir / "model.pth"
    torch.save(mock_model.state_dict(), model_path)
    
    # Initialize corruptor
    corruptor = ModelCorruptor(
        base_model_path=str(model_path),
        output_dir=str(test_data_dir),
        device="cpu"
    )
    
    assert corruptor.corruption_factors.shape == (12,)
    assert len(corruptor.cdiff) == 11

def test_noise_generation(test_data_dir, mock_model):
    """Test noise array generation."""
    model_path = test_data_dir / "model.pth"
    torch.save(mock_model.state_dict(), model_path)
    
    corruptor = ModelCorruptor(str(model_path), str(test_data_dir))
    corruptor.generate_noise()
    
    assert len(corruptor.noise_arrays) == sum(1 for _ in mock_model.parameters())
    for noise in corruptor.noise_arrays:
        assert isinstance(noise, torch.Tensor)
        assert not torch.isnan(noise).any()
        assert not torch.isinf(noise).any()

def test_model_corruption(test_data_dir, mock_model):
    """Test model corruption process."""
    model_path = test_data_dir / "model.pth"
    torch.save(mock_model.state_dict(), model_path)
    
    corruptor = ModelCorruptor(str(model_path), str(test_data_dir))
    corruptor.corrupt_model(seed=42)
    
    # Check output files
    corrupted_dir = test_data_dir / f"{model_path.stem}_corrupted_SEED_42"
    assert corrupted_dir.exists()
    
    # Check number of corrupted models
    corrupted_models = list(corrupted_dir.glob("*.pth"))
    assert len(corrupted_models) == 12

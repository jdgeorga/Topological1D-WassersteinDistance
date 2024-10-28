import os
from pathlib import Path
import torch
import numpy as np
from nequip.scripts.deploy import load_deployed_model
import shutil
import logging
import traceback

class ModelCorruptor:
    """Class for corrupting ML models with controlled noise."""
    
    def __init__(self, base_model_path: str, output_dir: str, device: str = "cpu", logger: logging.Logger = None):
        """
        Initialize the model corruptor.
        
        Args:
            base_model_path: Path to the base model file
            output_dir: Directory to save corrupted models
            device: Device to use for computations ("cpu" or "cuda")
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            # Validate inputs
            self.base_model_path = Path(base_model_path)
            if not self.base_model_path.exists():
                raise FileNotFoundError(f"Base model not found: {base_model_path}")
                
            self.output_dir = Path(output_dir)
            if not self.output_dir.parent.exists():
                raise FileNotFoundError(f"Parent directory does not exist: {output_dir}")
                
            if device not in ['cpu', 'cuda']:
                raise ValueError(f"Invalid device: {device}")
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = 'cpu'
            self.device = device
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load base model
            try:
                self.model, self.metadata = load_deployed_model(
                    model_path=self.base_model_path,
                    device=self.device,
                    freeze=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load base model: {str(e)}")
                
            # Generate corruption factors
            self.corruption_factors = np.logspace(-4.3, -0.3, 12)
            self.cdiff = np.diff(self.corruption_factors)
            
        except Exception as e:
            self.logger.error(f"Error initializing ModelCorruptor: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def generate_noise(self):
        """Generate noise arrays based on parameter standard deviations."""
        try:
            self.noise_arrays = []
            for param in self.model.parameters():
                param_std = torch.std(param)
                if torch.isnan(param_std) or torch.isinf(param_std):
                    raise ValueError(f"Invalid parameter standard deviation: {param_std}")
                    
                noise = torch.normal(0.0, param_std.item(), param.shape, device=self.device)
                if torch.isnan(noise).any() or torch.isinf(noise).any():
                    raise ValueError("Generated noise contains invalid values")
                    
                self.noise_arrays.append(noise)
                
        except Exception as e:
            self.logger.error(f"Error generating noise arrays: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def corrupt_model(self, seed: int):
        """
        Generate corrupted versions of the model with different noise levels.
        
        Args:
            seed: Random seed for reproducibility
        """
        try:
            # Set random seed
            torch.manual_seed(seed)
            
            # Generate noise arrays
            self.generate_noise()
            
            # Create directory for this seed
            model_name = self.base_model_path.stem
            corrupted_dir = self.output_dir / f"{model_name}_corrupted_SEED_{seed}"
            if corrupted_dir.exists():
                shutil.rmtree(corrupted_dir)
            corrupted_dir.mkdir()
            
            # Generate corrupted models
            for i in range(len(self.corruption_factors)):
                cfac_file = corrupted_dir / f"corruptfac_{self.corruption_factors[i]}_{i}.pth"
                
                # Remove existing file if present
                if cfac_file.exists():
                    os.remove(cfac_file)
                    
                # Calculate corruption factor
                if i == 0:
                    add_fac = self.corruption_factors[0]
                else:
                    add_fac = self.cdiff[i - 1]
                    
                # Apply corruption
                try:
                    for ct, param in enumerate(self.model.parameters()):
                        corrupted_param = param + add_fac * self.noise_arrays[ct]
                        
                        # Check for numerical issues
                        if torch.isnan(corrupted_param).any() or torch.isinf(corrupted_param).any():
                            raise ValueError(f"Invalid values in corrupted parameters at factor {add_fac}")
                            
                        param.data = corrupted_param
                        
                        # Log stats for first parameter
                        if ct == 0:
                            self.logger.info(
                                f"Corruption factor: {self.corruption_factors[i]}, "
                                f"Parameter std: {torch.std(param.data).item()}, "
                                f"Noise std: {torch.std(add_fac * self.noise_arrays[ct]).item()}"
                            )
                except Exception as e:
                    raise RuntimeError(f"Error applying corruption at factor {add_fac}: {str(e)}")
                    
                # Save corrupted model
                try:
                    torch.jit.save(self.model, cfac_file, _extra_files=self.metadata)
                except Exception as e:
                    raise RuntimeError(f"Failed to save corrupted model: {str(e)}")
                    
            # Verify outputs
            expected_files = len(self.corruption_factors)
            actual_files = len(list(corrupted_dir.glob("*.pth")))
            if actual_files != expected_files:
                raise RuntimeError(f"Expected {expected_files} output files, but found {actual_files}")
                
        except Exception as e:
            self.logger.error(f"Error corrupting model with seed {seed}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

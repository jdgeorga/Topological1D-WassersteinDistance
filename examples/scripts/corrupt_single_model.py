#!/usr/bin/env python
"""
Example script showing how to corrupt a single model.
"""
from pathlib import Path
from src.corruption.generate import ModelCorruptor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define paths
    base_model = "models/lmp_sw_mos2.pth"
    output_dir = "corrupted_models/example"
    
    # Initialize corruptor
    corruptor = ModelCorruptor(
        base_model_path=base_model,
        output_dir=output_dir,
        device="cpu"
    )
    
    # Generate corrupted models with seed 42
    corruptor.corrupt_model(seed=42)
    
    logger.info(f"Corrupted models saved to {output_dir}")

if __name__ == "__main__":
    main()

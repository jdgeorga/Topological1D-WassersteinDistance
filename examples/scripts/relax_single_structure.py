#!/usr/bin/env python
"""
Example script showing how to relax a single structure.
"""
from pathlib import Path
from ase.io import read
from src.relaxation.allegro import StructureOptimizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load structure
    atoms = read("structures/MoS2_WSe2_1D.xyz")
    
    # Set up models
    intralayer_models = {
        'layer1': "models/lmp_sw_mos2.pth",
        'layer2': "models/lmp_sw_wse2.pth"
    }
    interlayer_model = "models/lmp_kc_wse2_mos2.pth"
    
    # Initialize optimizer
    optimizer = StructureOptimizer(
        intralayer_models=intralayer_models,
        interlayer_model=interlayer_model,
        layer_symbols=[["Mo", "S", "S"], ["W", "Se", "Se"]]
    )
    
    # Relax structure
    relaxed_atoms, final_energy = optimizer.relax_structure(
        atoms,
        output_prefix="example_relaxation",
        initial_displacement=0.1
    )
    
    logger.info(f"Final energy: {final_energy:.3f} eV")

if __name__ == "__main__":
    main()

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory
from moirecompare.calculators import AllegroCalculator, NLayerCalculator
import logging
import traceback

class StructureOptimizer:
    """Class for optimizing atomic structures using Allegro calculators."""
    
    def __init__(
        self,
        intralayer_models: Dict[str, str],
        interlayer_model: str,
        layer_symbols: List[List[str]],
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the structure optimizer.
        
        Args:
            intralayer_models: Dict mapping layer names to model paths
            interlayer_model: Path to interlayer interaction model
            layer_symbols: List of atomic symbols for each layer
            device: Device to use for calculations
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            # Validate inputs
            if not all(Path(path).exists() for path in intralayer_models.values()):
                raise FileNotFoundError("One or more intralayer model files not found")
            if not Path(interlayer_model).exists():
                raise FileNotFoundError(f"Interlayer model file not found: {interlayer_model}")
            
            self.intralayer_models = {k: Path(v) for k, v in intralayer_models.items()}
            self.interlayer_model = Path(interlayer_model)
            self.layer_symbols = layer_symbols
            self.device = device
            
        except Exception as e:
            self.logger.error(f"Error initializing StructureOptimizer: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def setup_calculators(self, atoms: Atoms) -> NLayerCalculator:
        """
        Set up the combined calculator for structure optimization.
        
        Args:
            atoms: ASE Atoms object to optimize
            
        Returns:
            Combined NLayerCalculator
        """
        try:
            # Validate input
            if 'atom_types' not in atoms.arrays:
                raise ValueError("Atoms object missing 'atom_types' array")
            
            # Split atoms into layers
            at_1 = atoms[atoms.arrays['atom_types'] < 3]
            at_2 = atoms[atoms.arrays['atom_types'] >= 3]
            
            if len(at_1) == 0 or len(at_2) == 0:
                raise ValueError("Invalid layer splitting")
            
            # Set up intralayer calculators
            try:
                intra_calc_1 = AllegroCalculator(
                    at_1,
                    self.layer_symbols[0],
                    model_file=str(self.intralayer_models['layer1']),
                    device=self.device
                )
                intra_calc_2 = AllegroCalculator(
                    at_2,
                    self.layer_symbols[1],
                    model_file=str(self.intralayer_models['layer2']),
                    device=self.device
                )
            except Exception as e:
                self.logger.error("Failed to initialize intralayer calculators")
                raise
            
            # Set up interlayer calculator
            try:
                inter_calc = AllegroCalculator(
                    atoms,
                    self.layer_symbols,
                    model_file=str(self.interlayer_model),
                    device=self.device
                )
            except Exception as e:
                self.logger.error("Failed to initialize interlayer calculator")
                raise
            
            # Combine calculators
            return NLayerCalculator(
                atoms,
                [intra_calc_1, intra_calc_2],
                [inter_calc],
                self.layer_symbols,
                device=self.device
            )
            
        except Exception as e:
            self.logger.error(f"Error setting up calculators: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def relax_structure(
        self,
        atoms: Atoms,
        output_prefix: str,
        fmax: float = 5e-3,
        steps: int = 1000,
        maxstep: float = 0.05,
        initial_displacement: Optional[float] = None
    ) -> Tuple[Atoms, float]:
        """
        Relax atomic structure using FIRE optimizer.
        
        Args:
            atoms: Structure to optimize
            output_prefix: Prefix for output files
            fmax: Maximum force criterion
            steps: Maximum optimization steps
            maxstep: Maximum step size
            initial_displacement: Optional initial displacement magnitude
            
        Returns:
            Tuple of (optimized structure, final energy)
        """
        try:
            # Validate inputs
            if not isinstance(atoms, Atoms):
                raise TypeError("Input must be ASE Atoms object")
            if fmax <= 0:
                raise ValueError(f"Invalid fmax value: {fmax}")
            if steps <= 0:
                raise ValueError(f"Invalid steps value: {steps}")
            if maxstep <= 0:
                raise ValueError(f"Invalid maxstep value: {maxstep}")
            
            # Apply initial displacement if specified
            if initial_displacement is not None:
                if initial_displacement <= 0:
                    raise ValueError(f"Invalid initial_displacement value: {initial_displacement}")
                atoms.rattle(stdev=initial_displacement)
            
            # Set up calculator
            atoms.calc = self.setup_calculators(atoms)
            
            # Calculate initial energy
            try:
                atoms.calc.calculate(atoms)
                initial_energy = atoms.calc.results['energy']
                self.logger.info(f"Initial energy: {initial_energy:.3f} eV")
            except Exception as e:
                self.logger.error("Failed to calculate initial energy")
                raise
            
            # Set up optimizer
            try:
                dyn = FIRE(
                    atoms,
                    trajectory=f"{output_prefix}.traj",
                    maxstep=maxstep
                )
            except Exception as e:
                self.logger.error("Failed to initialize optimizer")
                raise
            
            # Run optimization
            try:
                dyn.run(fmax=fmax, steps=steps)
            except Exception as e:
                self.logger.error("Error during optimization")
                raise
            
            # Get final energy
            final_energy = atoms.calc.results['energy']
            self.logger.info(f"Final energy: {final_energy:.3f} eV")
            
            # Save trajectory and lowest energy structure
            try:
                self._save_trajectory(output_prefix)
            except Exception as e:
                self.logger.error("Failed to save trajectory")
                raise
            
            return atoms, final_energy
            
        except Exception as e:
            self.logger.error(f"Error relaxing structure: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def _save_trajectory(self, output_prefix: str):
        """Save trajectory and lowest energy structure."""
        try:
            # Read trajectory
            traj = Trajectory(f"{output_prefix}.traj")
            
            # Find lowest energy image
            lowest_energy_image = min(traj, key=lambda atoms: atoms.get_potential_energy())
            
            # Save structures
            write(f"{output_prefix}_lowest_energy.xyz", lowest_energy_image, format="extxyz")
            write(f"{output_prefix}.traj.xyz", [atom for atom in traj], format="extxyz")
            
        except Exception as e:
            self.logger.error(f"Error saving trajectory: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

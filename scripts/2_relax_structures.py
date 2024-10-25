#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np
from ase.io import read
import torch
import traceback
from src.relaxation.allegro import StructureOptimizer
from src.utils.logging import setup_logger

def validate_args(args, logger):
    """Validate command line arguments."""
    try:
        # Check structure file
        structure_path = Path(args.structure)
        if not structure_path.exists():
            logger.error(f"Structure file not found: {structure_path}")
            return False
            
        # Validate output directory
        output_dir = Path(args.output_dir)
        if not output_dir.parent.exists():
            logger.error(f"Parent directory does not exist: {output_dir.parent}")
            return False
            
        # Check seed and corruption index
        if args.seed < 0:
            logger.error(f"Invalid seed value: {args.seed}")
            return False
            
        if not 0 <= args.corruption_idx < 12:
            logger.error(f"Corruption index must be between 0 and 11, got {args.corruption_idx}")
            return False
            
        # Validate device
        if args.device not in ['cpu', 'cuda']:
            logger.error(f"Invalid device: {args.device}")
            return False
            
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'
            
        return True
    except Exception as e:
        logger.error(f"Error validating arguments: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def validate_models(corruption_factors, seed, models_dir, logger):
    """Validate that all required model files exist."""
    try:
        required_models = {
            'layer1': f"lmp_sw_mos2_corrupted_SEED_{seed}",
            'layer2': f"lmp_sw_wse2_corrupted_SEED_{seed}",
            'interlayer': f"lmp_kc_wse2_mos2_corrupted_SEED_{seed}"
        }
        
        for model_type, model_dir in required_models.items():
            model_path = models_dir / model_dir / f"corruptfac_{corruption_factors[args.corruption_idx]}_{args.corruption_idx}.pth"
            if not model_path.exists():
                logger.error(f"Required model file not found: {model_path}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating model files: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def check_system_resources(logger):
    """Check system resources before starting."""
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(f"High memory usage: {memory.percent}%")
            
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent}%")
            
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            logger.warning(f"Low disk space: {disk.free / (2**30):.1f} GB free")
            
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Relax structures with corrupted models")
    parser.add_argument("--structure", type=str, required=True, help="Input structure file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, required=True, help="Model seed")
    parser.add_argument("--corruption-idx", type=int, required=True, help="Corruption factor index")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        f"relax_struct_{args.seed}_{args.corruption_idx}",
        log_dir=Path("logs/relaxation")
    )
    
    try:
        logger.info(f"Starting structure relaxation for seed {args.seed}, corruption index {args.corruption_idx}")
        
        # Validate arguments
        if not validate_args(args, logger):
            return 1
            
        # Check system resources
        if not check_system_resources(logger):
            return 1
            
        # Load structure
        try:
            logger.info(f"Loading structure from {args.structure}")
            atoms = read(args.structure)
        except Exception as e:
            logger.error(f"Failed to load structure: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Set up model paths
        corruption_factors = np.logspace(-4.3, -0.3, 12)
        models_dir = Path("corrupted_models")
        
        # Validate model files
        if not validate_models(corruption_factors, args.seed, models_dir, logger):
            return 1
            
        # Set up model paths
        intralayer_models = {
            'layer1': models_dir / f"lmp_sw_mos2_corrupted_SEED_{args.seed}" / 
                     f"corruptfac_{corruption_factors[args.corruption_idx]}_{args.corruption_idx}.pth",
            'layer2': models_dir / f"lmp_sw_wse2_corrupted_SEED_{args.seed}" / 
                     f"corruptfac_{corruption_factors[args.corruption_idx]}_{args.corruption_idx}.pth"
        }
        interlayer_model = models_dir / f"lmp_kc_wse2_mos2_corrupted_SEED_{args.seed}" / \
                          f"corruptfac_{corruption_factors[args.corruption_idx]}_{args.corruption_idx}.pth"
                          
        # Initialize optimizer
        try:
            logger.info("Initializing structure optimizer...")
            optimizer = StructureOptimizer(
                intralayer_models=intralayer_models,
                interlayer_model=interlayer_model,
                layer_symbols=[["Mo", "S", "S"], ["W", "Se", "Se"]],
                device=args.device
            )
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Relax structure
        output_prefix = output_dir / f"relaxed_SEED_{args.seed}_idx_{args.corruption_idx}"
        try:
            logger.info("Starting structure relaxation...")
            optimizer.relax_structure(
                atoms,
                output_prefix=str(output_prefix),
                initial_displacement=0.1
            )
        except Exception as e:
            logger.error(f"Error during structure relaxation: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Verify outputs
        expected_files = [
            output_prefix.with_suffix('.traj'),
            output_prefix.with_suffix('.traj.xyz'),
            output_prefix.with_name(f"{output_prefix.stem}_lowest_energy.xyz")
        ]
        
        for file in expected_files:
            if not file.exists():
                logger.error(f"Expected output file not found: {file}")
                return 1
                
        logger.info("Structure relaxation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

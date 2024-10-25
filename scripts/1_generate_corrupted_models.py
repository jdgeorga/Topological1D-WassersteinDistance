#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import torch
import traceback
from src.corruption.generate import ModelCorruptor
from src.utils.logging import setup_logger

def validate_args(args, logger):
    """Validate command line arguments."""
    try:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
            
        output_dir = Path(args.output_dir)
        if not output_dir.parent.exists():
            logger.error(f"Parent directory does not exist: {output_dir.parent}")
            return False
            
        if args.seed < 0:
            logger.error(f"Invalid seed value: {args.seed}")
            return False
            
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
    parser = argparse.ArgumentParser(description="Generate corrupted ML models")
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        f"corrupt_model_{args.seed}",
        log_dir=Path("logs/corruption")
    )
    
    try:
        logger.info(f"Starting model corruption for seed {args.seed}")
        
        # Validate arguments
        if not validate_args(args, logger):
            return 1
            
        # Check system resources
        if not check_system_resources(logger):
            return 1
            
        # Initialize corruptor
        logger.info("Initializing model corruptor...")
        corruptor = ModelCorruptor(
            base_model_path=args.model,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # Generate corrupted models
        logger.info("Generating corrupted models...")
        try:
            corruptor.corrupt_model(seed=args.seed)
        except Exception as e:
            logger.error(f"Error during model corruption: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Verify outputs
        output_dir = Path(args.output_dir)
        expected_files = 12  # One for each corruption factor
        actual_files = len(list(output_dir.glob(f"*SEED_{args.seed}/*.pth")))
        
        if actual_files != expected_files:
            logger.error(f"Expected {expected_files} output files, but found {actual_files}")
            return 1
            
        logger.info("Model corruption completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

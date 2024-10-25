#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import shutil
import hashlib
from src.utils.logging import setup_logger

# Define expected files and their MD5 hashes
EXPECTED_FILES = {
    'structures/MoS2_WSe2_1D.xyz': 'hash_value_1',
    'structures/MoS2_WSe2_2D.xyz': 'hash_value_2',
    'potentials/lmp_sw_mos2.pth': 'hash_value_3',
    'potentials/lmp_sw_wse2.pth': 'hash_value_4',
    'potentials/lmp_kc_wse2_mos2.pth': 'hash_value_5',
    'style/matplotlib.rc': 'hash_value_6'
}

def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def setup_directories(base_dir: Path, logger) -> bool:
    """Create directory structure."""
    try:
        # Create input directories
        input_dirs = [
            base_dir / 'data/input/structures',
            base_dir / 'data/input/potentials',
            base_dir / 'data/input/style'
        ]
        
        # Create output directories
        output_dirs = [
            base_dir / 'data/output/corrupted_models',
            base_dir / 'data/output/relaxed_structures',
            base_dir / 'data/output/distances',
            base_dir / 'data/output/plots'
        ]
        
        for directory in input_dirs + output_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

def verify_input_files(base_dir: Path, logger) -> bool:
    """Verify presence and integrity of input files."""
    input_dir = base_dir / 'data/input'
    success = True
    
    for file_path, expected_hash in EXPECTED_FILES.items():
        full_path = input_dir / file_path
        if not full_path.exists():
            logger.error(f"Missing input file: {file_path}")
            success = False
            continue
            
        actual_hash = calculate_md5(full_path)
        if actual_hash != expected_hash:
            logger.error(f"Hash mismatch for {file_path}")
            logger.error(f"Expected: {expected_hash}")
            logger.error(f"Got: {actual_hash}")
            success = False
            
    return success

def main():
    parser = argparse.ArgumentParser(description="Set up data directory structure")
    parser.add_argument("--base-dir", type=str, default=".",
                       help="Base directory for the project")
    args = parser.parse_args()
    
    logger = setup_logger("setup_data", log_dir="logs")
    base_dir = Path(args.base_dir)
    
    try:
        # Create directory structure
        if not setup_directories(base_dir, logger):
            return 1
            
        # Verify input files
        if not verify_input_files(base_dir, logger):
            logger.error("Input file verification failed")
            return 1
            
        logger.info("Data directory setup completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

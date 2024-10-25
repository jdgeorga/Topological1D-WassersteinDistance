#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np
from ase.io import read
import multiprocessing as mp
import traceback
from src.metrics.voronoi import VoronoiAnalyzer
from src.metrics.wasserstein import WassersteinMetric
from src.utils.logging import setup_logger

def validate_args(args, logger):
    """Validate command line arguments."""
    try:
        # Check relaxed directory
        relaxed_dir = Path(args.relaxed_dir)
        if not relaxed_dir.exists():
            logger.error(f"Relaxed directory not found: {relaxed_dir}")
            return False
            
        # Check reference structure
        reference_path = Path(args.reference)
        if not reference_path.exists():
            logger.error(f"Reference structure not found: {reference_path}")
            return False
            
        # Check output directory
        output_path = Path(args.output).parent
        if not output_path.exists():
            logger.error(f"Output directory does not exist: {output_path}")
            return False
            
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
            
        # Check available CPUs
        n_cpus = mp.cpu_count()
        logger.info(f"Available CPUs: {n_cpus}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def process_structure(args, logger):
    """Process a single structure and calculate distances."""
    structure_path, reference_path, lattice_vectors = args
    
    try:
        # Load structures
        logger.debug(f"Loading structure: {structure_path}")
        structure = read(structure_path)
        reference = read(reference_path)
        
        # Initialize analyzers
        voronoi = VoronoiAnalyzer()
        wasserstein = WassersteinMetric(lattice_vectors)
        
        # Get primitive cell
        pristine_cell = voronoi.get_primitive_voronoi_cell(lattice_vectors)
        if pristine_cell is None:
            logger.error(f"Failed to get primitive cell for {structure_path}")
            return None
        
        # Calculate displacements
        structure_points = structure.positions[structure.arrays['atom_types'] == 0, :2]
        reference_points = reference.positions[reference.arrays['atom_types'] == 0, :2]
        query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]
        
        # Pad periodic images
        padded_structure = voronoi.pad_periodic_image(structure_points, structure.cell[:2, :2])
        padded_reference = voronoi.pad_periodic_image(reference_points, reference.cell[:2, :2])
        
        # Calculate interpolated points
        try:
            interpolated_points = voronoi.interpolate_displacements(
                padded_structure,
                padded_reference,
                query_points,
                pristine_cell
            )
        except Exception as e:
            logger.error(f"Error calculating interpolated points for {structure_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
        
        # Calculate Wasserstein distance
        try:
            distance, _, _, _ = wasserstein.calculate_distance(
                interpolated_points[0],
                interpolated_points[1]
            )
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance for {structure_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
        
        return distance
        
    except Exception as e:
        logger.error(f"Error processing structure {structure_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate structural distances")
    parser.add_argument("--relaxed-dir", type=str, required=True, help="Directory with relaxed structures")
    parser.add_argument("--reference", type=str, required=True, help="Reference structure")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        "calculate_distances",
        log_dir=Path("logs/distances")
    )
    
    try:
        logger.info("Starting distance calculations")
        
        # Validate arguments
        if not validate_args(args, logger):
            return 1
            
        # Check system resources
        if not check_system_resources(logger):
            return 1
            
        # Load reference structure for lattice vectors
        try:
            reference = read(args.reference)
            lattice_vectors = np.array([[3.1841, 0.0], [-1.5920, 2.7575]])
        except Exception as e:
            logger.error(f"Failed to load reference structure: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Get all relaxed structures
        relaxed_dir = Path(args.relaxed_dir)
        structure_paths = list(relaxed_dir.glob("*_lowest_energy.xyz"))
        
        if not structure_paths:
            logger.error(f"No structures found in {relaxed_dir}")
            return 1
            
        logger.info(f"Found {len(structure_paths)} structures to process")
        
        # Prepare arguments for parallel processing
        process_args = [(str(path), args.reference, lattice_vectors) for path in structure_paths]
        
        # Calculate distances in parallel
        n_cpus = mp.cpu_count()
        logger.info(f"Using {n_cpus} CPUs for parallel processing")
        
        with mp.Pool() as pool:
            distances = pool.map(process_structure, process_args)
        
        # Check for failed calculations
        failed_calcs = [i for i, d in enumerate(distances) if d is None]
        if failed_calcs:
            logger.error(f"Failed calculations for structures: {failed_calcs}")
            return 1
        
        # Save results
        try:
            results = np.column_stack([
                [int(p.stem.split('_')[2]) for p in structure_paths],  # Seeds
                [int(p.stem.split('_')[4]) for p in structure_paths],  # Corruption indices
                distances
            ])
            np.save(args.output, results)
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

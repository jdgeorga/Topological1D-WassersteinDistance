#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import itertools
import traceback
from src.utils.logging import setup_logger

def validate_args(args, logger):
    """Validate command line arguments."""
    try:
        # Check input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False
            
        # Check output directory parent exists
        output_dir = Path(args.output_dir)
        if not output_dir.parent.exists():
            logger.error(f"Parent directory does not exist: {output_dir.parent}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating arguments: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def validate_input_files(input_dir, logger):
    """Validate that all required input files exist."""
    try:
        required_structures = ["1D", "2D"]
        required_types = ["interlayer", "intralayer"]
        
        for struct in required_structures:
            for analysis_type in required_types:
                input_file = input_dir / f"MoS2_WSe2_{struct}" / analysis_type / "distances.npy"
                if not input_file.exists():
                    logger.error(f"Required input file not found: {input_file}")
                    return False
                    
                # Check file is not empty
                if input_file.stat().st_size == 0:
                    logger.error(f"Input file is empty: {input_file}")
                    return False
                    
        return True
    except Exception as e:
        logger.error(f"Error validating input files: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def process_distances(distances_array, logger):
    """
    Process raw distances into a format suitable for plotting.
    
    Args:
        distances_array: Array with columns [seed, corruption_idx, distance]
        logger: Logger instance for error reporting
        
    Returns:
        Processed array with averaged values and proper grouping
    """
    def average_over_relaxations(group):
        """Average distances over repeated relaxations."""
        group_array = np.array(group)
        return [
            *group_array[0, :2],  # Keep corruption_idx and seed
            np.mean(group_array[:, 2].astype(float))  # Average distances
        ]

    try:
        # Sort by corruption factor and seed
        sorted_data = sorted(distances_array, key=lambda x: (x[1], x[0]))
        
        # Group by corruption factor
        grouped_data = defaultdict(list)
        for key, group in itertools.groupby(sorted_data, key=lambda x: x[1]):
            group_list = list(group)
            averaged = average_over_relaxations(group_list)
            grouped_data[int(key)].append(averaged)

        # Check for missing seeds and add averaged values
        for corruption_factor in range(12):
            existing_seeds = set(int(row[0]) for row in grouped_data[corruption_factor])
            missing_seeds = set(range(10)) - existing_seeds
            
            if missing_seeds:
                logger.warning(f"Missing seeds {missing_seeds} for corruption factor {corruption_factor}")
                avg_distance = np.mean([row[2] for row in grouped_data[corruption_factor]])
                for seed in missing_seeds:
                    new_row = [seed, corruption_factor, avg_distance]
                    grouped_data[corruption_factor].append(new_row)

        # Convert to array and sort
        final_data = [item for sublist in grouped_data.values() for item in sublist]
        final_array = np.array(sorted(final_data, key=lambda x: (x[1], x[0])))
        
        return final_array
        
    except Exception as e:
        logger.error(f"Error processing distances: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Process calculated distances")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with raw distances")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        "process_distances",
        log_dir=Path("logs/analysis")
    )
    
    try:
        logger.info("Starting distance processing")
        
        # Validate arguments
        if not validate_args(args, logger):
            return 1
            
        # Validate input files
        if not validate_input_files(Path(args.input_dir), logger):
            return 1
            
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each type of distance
        for struct_type in ["1D", "2D"]:
            for analysis_type in ["interlayer", "intralayer"]:
                logger.info(f"Processing {struct_type} {analysis_type} distances")
                
                try:
                    input_file = Path(args.input_dir) / f"MoS2_WSe2_{struct_type}" / analysis_type / "distances.npy"
                    raw_distances = np.load(input_file)
                    
                    processed_distances = process_distances(raw_distances, logger)
                    if processed_distances is None:
                        logger.error(f"Failed to process distances for {struct_type} {analysis_type}")
                        continue
                        
                    # Save processed results
                    output_file = output_dir / f"processed_{analysis_type}_distances_{struct_type}.npy"
                    np.save(output_file, processed_distances)
                    logger.info(f"Saved processed results to {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing {struct_type} {analysis_type}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

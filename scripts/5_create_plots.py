#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import numpy as np
import traceback
from src.visualization.plots import DistancePlotter
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
            
        # Check style file if provided
        if args.style and not Path(args.style).exists():
            logger.error(f"Style file not found: {args.style}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating arguments: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def validate_input_files(input_dir, logger):
    """Validate that all required input files exist."""
    try:
        required_files = [
            f"processed_{analysis_type}_distances_{dim}.npy"
            for dim in ["1D", "2D"]
            for analysis_type in ["interlayer", "intralayer"]
        ]
        
        for file in required_files:
            file_path = input_dir / file
            if not file_path.exists():
                logger.error(f"Required input file not found: {file_path}")
                return False
                
            # Check file is not empty
            if file_path.stat().st_size == 0:
                logger.error(f"Input file is empty: {file_path}")
                return False
                
            # Try loading the file
            try:
                data = np.load(file_path)
                if data.size == 0:
                    logger.error(f"Input file contains no data: {file_path}")
                    return False
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating input files: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def load_distances(input_dir, logger):
    """Load all distance data files."""
    try:
        distances = {
            "1D": {}, "2D": {}
        }
        
        for dim in ["1D", "2D"]:
            for analysis_type in ["interlayer", "intralayer"]:
                file_path = input_dir / f"processed_{analysis_type}_distances_{dim}.npy"
                try:
                    distances[dim][analysis_type] = np.load(file_path)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    return None
                    
        return distances
    except Exception as e:
        logger.error(f"Error loading distances: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def create_plots(plotter, distances, output_dir, logger):
    """Create all plots."""
    try:
        # Create individual plots
        for dim in ["1D", "2D"]:
            for dist_type in ["interlayer", "intralayer"]:
                logger.info(f"Creating {dist_type} plot for {dim}")
                
                try:
                    dist_data = distances[dim][dist_type]
                    corruption_indices = dist_data[:, 1].astype(int)
                    dist_values = dist_data[:, 2]
                    
                    output_path = output_dir / f"{dist_type}_distances_{dim}.pdf"
                    plotter.plot_distance_vs_corruption(
                        dist_values,
                        output_path=str(output_path),
                        title=f"{dist_type.capitalize()} Distances - {dim}",
                        gradient_thresholds={"threshold": 1e-2}
                    )
                    logger.info(f"Saved plot to {output_path}")
                except Exception as e:
                    logger.error(f"Error creating plot for {dim} {dist_type}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue
        
        # Create comparison plots
        # 1D vs 2D comparison
        for dist_type in ["interlayer", "intralayer"]:
            try:
                output_path = output_dir / f"{dist_type}_1D_vs_2D_comparison.pdf"
                plotter.plot_distance_comparison(
                    distances["1D"][dist_type][:, 2],
                    distances["2D"][dist_type][:, 2],
                    output_path=str(output_path),
                    labels=(f"1D {dist_type}", f"2D {dist_type}"),
                    title=f"1D vs 2D {dist_type.capitalize()} Distance Comparison"
                )
                logger.info(f"Saved comparison plot to {output_path}")
            except Exception as e:
                logger.error(f"Error creating comparison plot for {dist_type}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
        
        # Interlayer vs Intralayer comparison
        for dim in ["1D", "2D"]:
            try:
                output_path = output_dir / f"{dim}_interlayer_vs_intralayer_comparison.pdf"
                plotter.plot_distance_comparison(
                    distances[dim]["interlayer"][:, 2],
                    distances[dim]["intralayer"][:, 2],
                    output_path=str(output_path),
                    labels=("Interlayer", "Intralayer"),
                    title=f"{dim} Interlayer vs Intralayer Distance Comparison"
                )
                logger.info(f"Saved layer comparison plot to {output_path}")
            except Exception as e:
                logger.error(f"Error creating layer comparison plot for {dim}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
                
        return True
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Create publication plots")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with processed distances")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--style", type=str, help="Path to matplotlib style file")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        "create_plots",
        log_dir=Path("logs/plotting")
    )
    
    try:
        logger.info("Starting plot creation")
        
        # Validate arguments
        if not validate_args(args, logger):
            return 1
            
        # Validate input files
        if not validate_input_files(Path(args.input_dir), logger):
            return 1
            
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load distances
        distances = load_distances(Path(args.input_dir), logger)
        if distances is None:
            return 1
            
        # Initialize plotter
        try:
            plotter = DistancePlotter(style_path=args.style)
        except Exception as e:
            logger.error(f"Error initializing plotter: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
            
        # Create plots
        if not create_plots(plotter, distances, output_dir, logger):
            return 1
            
        logger.info("Plot creation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

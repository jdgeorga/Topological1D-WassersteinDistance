# Topological1D-WassersteinDistance: Wasserstein Distance Analysis for 2D Materials

This repository contains the code for analyzing structural changes in 2D materials using Wasserstein distances and model corruption studies.

## Overview

The workflow analyzes how structural changes in MoS2-WSe2 heterostructures correlate with systematic corruptions of machine learning interatomic potentials (MLIPs). It includes:

1. Model corruption with controlled noise
2. Structure relaxation using corrupted models
3. Distance calculations using Wasserstein metrics
4. Analysis and visualization of results

## Repository Structure

dw-distance/
├── data/                      # Data directory
│   ├── input/                # Input files
│   │   ├── structures/      # Initial atomic structures
│   │   ├── potentials/      # ML potential models
│   │   └── style/          # Plotting style files
│   └── output/              # Generated data
├── src/                      # Source code
│   ├── corruption/          # Model corruption tools
│   ├── relaxation/          # Structure relaxation
│   ├── metrics/             # Distance metrics
│   ├── visualization/       # Plotting tools
│   └── utils/              # Utility functions
└── scripts/                  # Analysis scripts

## Prerequisites

- Python 3.9+
- PyTorch 2.0.1s+
- ASE (Atomic Simulation Environment)
- NequIP
- MoireCompare
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone the repository:
git clone https://github.com/username/dw-distance.git
cd dw-distance

2. Set up the environment:
python -m venv dw-distance
source dw-distance/bin/activate
pip install -r requirements.txt

3. Set up the data directory:
python scripts/setup_data.py

## Usage

The complete workflow can be run using the master script:

./run_workflow.sh --account YOUR_ACCOUNT --partition YOUR_PARTITION

Or run individual steps:

# Generate corrupted models
./scripts/1_run_corruption.sh

# Relax structures
./scripts/2_run_relaxation.sh

# Calculate distances
./scripts/3_run_distance_calculation.sh

# Analyze results and create plots
./scripts/4_run_analysis.sh

### Command Line Options

The master workflow script accepts several options:

- --start-step N: Start from step N (default: 1)
- --end-step N: End at step N (default: 4)
- --account: SLURM account name
- --partition: SLURM partition name

## Input Files

### Required Structure Files
- data/input/structures/MoS2_WSe2_1D.xyz: 1D heterostructure
- data/input/structures/MoS2_WSe2_2D.xyz: 2D heterostructure

### Required Model Files
- data/input/potentials/lmp_sw_mos2.pth: MoS2 intralayer model
- data/input/potentials/lmp_sw_wse2.pth: WSe2 intralayer model
- data/input/potentials/lmp_kc_wse2_mos2.pth: Interlayer interaction model

## Output Files

The workflow generates several types of output:

1. Corrupted Models
   - Located in data/output/corrupted_models/
   - One directory per base model, containing variants with different corruption levels

2. Relaxed Structures
   - Located in data/output/relaxed_structures/
   - XYZ files of relaxed structures for each corruption level

3. Distance Calculations
   - Located in data/output/distances/
   - NumPy arrays containing Wasserstein distances

4. Analysis Results
   - Located in data/output/plots/
   - Publication-quality figures showing correlations and trends

## Error Handling

The code includes comprehensive error handling and logging:

- All errors are logged to logs/ directory
- Each run creates a timestamped log file
- Detailed error messages and stack traces are preserved
- Resource usage is monitored and warnings are issued

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

@article{Georgaras2024,
    title={Model corruption and distance metrics for 2D materials},
    author={Georgaras, J.D., Ramdas, A., and da Jornada, F. H.},
    journal={Nature Communications},
    year={2024}
}

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Johnathan Dimitrios Georgaras (jdgeorga@stanford.edu)

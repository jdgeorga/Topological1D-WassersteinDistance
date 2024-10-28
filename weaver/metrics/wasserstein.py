from typing import Tuple, List, Optional
import numpy as np
from sklearn.neighbors import KernelDensity
import ot
import logging
import traceback

class WassersteinMetric:
    """Class for calculating Wasserstein distances between atomic configurations."""
    
    def __init__(self, lattice_vectors: np.ndarray, logger: Optional[logging.Logger] = None):
        """
        Initialize Wasserstein metric calculator.
        
        Args:
            lattice_vectors: 2x2 array of lattice vectors
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            if lattice_vectors.shape != (2, 2):
                raise ValueError(f"Invalid lattice vectors shape: {lattice_vectors.shape}")
            self.lattice_vectors = lattice_vectors
            self.cell_area = np.abs(np.linalg.det(lattice_vectors))
            
        except Exception as e:
            self.logger.error(f"Error initializing WassersteinMetric: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def calculate_distance(
        self,
        displacements1: np.ndarray,
        displacements2: np.ndarray,
        bandwidth: Optional[float] = None
    ) -> Tuple[float, np.ndarray, float, np.ndarray]:
        """
        Calculate Wasserstein distance between two sets of displacements.
        
        Args:
            displacements1: First set of displacements
            displacements2: Second set of displacements
            bandwidth: KDE bandwidth (default: calculated from data)
            
        Returns:
            Tuple of (distance, transport matrix,
                     uniform distance, uniform transport matrix)
        """
        try:
            # Validate inputs
            if displacements1.size == 0 or displacements2.size == 0:
                raise ValueError("Empty displacement arrays")
            if displacements1.shape[-1] != 2 or displacements2.shape[-1] != 2:
                raise ValueError("Displacements must be 2D")
                
            # Calculate bandwidth if not provided
            if bandwidth is None:
                bandwidth = np.linalg.norm(self.lattice_vectors[0]) / np.sqrt(4 * len(displacements1))
                
            # Calculate densities
            try:
                density1 = self._calculate_density(displacements1, bandwidth)
                density2 = self._calculate_density(displacements2, bandwidth)
            except Exception as e:
                self.logger.error(f"Error calculating densities: {str(e)}")
                self.logger.debug(traceback.format_exc())
                raise
                
            # Normalize densities
            density1_norm = density1 / np.sum(density1)
            density2_norm = density2 / np.sum(density2)
            
            # Create uniform densities
            density1_uniform = np.ones_like(density1_norm) / len(density1_norm)
            density2_uniform = np.ones_like(density2_norm) / len(density2_norm)
            
            # Calculate cost matrix
            try:
                cost_matrix = self._calculate_cost_matrix(displacements1, displacements2)
            except Exception as e:
                self.logger.error(f"Error calculating cost matrix: {str(e)}")
                self.logger.debug(traceback.format_exc())
                raise
                
            # Calculate optimal transport
            try:
                transport_matrix = ot.emd(density1_norm, density2_norm, cost_matrix)
                transport_matrix_uniform = ot.emd(density1_uniform, density2_uniform, cost_matrix)
            except Exception as e:
                self.logger.error(f"Error calculating optimal transport: {str(e)}")
                self.logger.debug(traceback.format_exc())
                raise
                
            # Calculate distances
            distance = np.sum(transport_matrix * cost_matrix)
            distance_uniform = np.sum(transport_matrix_uniform * cost_matrix)
            
            return distance, transport_matrix, distance_uniform, transport_matrix_uniform
            
        except Exception as e:
            self.logger.error(f"Error calculating Wasserstein distance: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def _calculate_density(
        self,
        points: np.ndarray,
        bandwidth: float
    ) -> np.ndarray:
        """
        Calculate KDE density for points.
        
        Args:
            points: Points to calculate density for
            bandwidth: KDE bandwidth
            
        Returns:
            Array of density values
        """
        try:
            kde = KernelDensity(
                bandwidth=bandwidth,
                metric='euclidean',
                kernel='gaussian'
            )
            
            kde.fit(points)
            log_density = kde.score_samples(points)
            density = np.exp(log_density)
            
            # Normalize by number of points
            normalization_factor = len(points) / np.sum(density)
            return density * normalization_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating density: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def _calculate_cost_matrix(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cost matrix with periodic boundary conditions.
        
        Args:
            points1: First set of points
            points2: Second set of points
            
        Returns:
            Cost matrix
        """
        try:
            # Create periodic images
            offsets = np.array([
                i * self.lattice_vectors[0] + j * self.lattice_vectors[1]
                for i in range(-2, 3)
                for j in range(-2, 3)
            ])
            
            # Calculate distances with periodic boundary conditions
            diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
            diff_periodic = diff[np.newaxis, :, :, :] + offsets[:, np.newaxis, np.newaxis, :]
            
            # Check for numerical issues
            if np.any(np.isnan(diff_periodic)) or np.any(np.isinf(diff_periodic)):
                raise ValueError("Invalid values in periodic differences")
                
            return np.min(np.linalg.norm(diff_periodic, axis=-1), axis=0)
            
        except Exception as e:
            self.logger.error(f"Error calculating cost matrix: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

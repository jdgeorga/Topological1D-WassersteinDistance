from typing import Tuple, List, Optional, Union
import numpy as np
from scipy.spatial import Voronoi, cKDTree, Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
import logging
import traceback

class VoronoiAnalyzer:
    """Class for analyzing atomic structures using Voronoi tessellation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Voronoi analyzer.
        
        Args:
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
    @staticmethod
    def get_primitive_voronoi_cell(lattice_vectors: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate the primitive Voronoi cell for given lattice vectors.
        
        Args:
            lattice_vectors: 2x2 array of lattice vectors
            
        Returns:
            Array of Voronoi cell vertices or None if generation fails
        """
        try:
            x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
            points = np.column_stack((x.ravel(), y.ravel()))
            lattice_points = points @ lattice_vectors
            
            vor = Voronoi(lattice_points)
            central_point_index = 4  # Center point in 3x3 grid
            central_region = vor.regions[vor.point_region[central_point_index]]
            
            if -1 not in central_region:
                return vor.vertices[central_region]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating primitive Voronoi cell: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def pad_periodic_image(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        n_a1: int = 2,
        n_a2: int = 2
    ) -> Optional[np.ndarray]:
        """
        Create periodic images of atomic positions.
        
        Args:
            positions: Atomic positions
            cell: Unit cell vectors
            n_a1: Number of repeats in a1 direction
            n_a2: Number of repeats in a2 direction
            
        Returns:
            Array of padded positions or None if padding fails
        """
        try:
            # Validate inputs
            if positions.size == 0:
                raise ValueError("Empty positions array")
            if cell.shape != (2, 2):
                raise ValueError(f"Invalid cell shape: {cell.shape}")
            if n_a1 < 1 or n_a2 < 1:
                raise ValueError(f"Invalid repeat numbers: n_a1={n_a1}, n_a2={n_a2}")
                
            i_range = np.concatenate((np.arange(0, n_a1 + 1), np.arange(-n_a1, 0)))
            j_range = np.concatenate((np.arange(0, n_a2 + 1), np.arange(-n_a2, 0)))
            i, j = np.meshgrid(i_range, j_range)
            i, j = i.flatten(), j.flatten()
            
            offsets = i[:, np.newaxis] * cell[0] + j[:, np.newaxis] * cell[1]
            padded_pos = positions[np.newaxis, :, :] + offsets[:, np.newaxis, :]
            
            return padded_pos.reshape(-1, positions.shape[-1])
            
        except Exception as e:
            self.logger.error(f"Error padding periodic images: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def interpolate_displacements(
        self,
        relaxed_points: np.ndarray,
        unrelaxed_points: np.ndarray,
        query_points: np.ndarray,
        pristine_cell: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Interpolate atomic displacements using Voronoi tessellation.
        
        Args:
            relaxed_points: Positions after relaxation
            unrelaxed_points: Initial positions
            query_points: Points to interpolate
            pristine_cell: Vertices of pristine Voronoi cell
            
        Returns:
            Tuple of (interpolated points in pristine cell,
                     interpolated points in unrelaxed structure,
                     unrelaxed circumcenters,
                     interpolated circumcenters)
            or None if interpolation fails
        """
        try:
            # Validate inputs
            if any(arr.size == 0 for arr in [relaxed_points, unrelaxed_points, query_points, pristine_cell]):
                raise ValueError("Empty input array")
            if not all(arr.shape[-1] == 2 for arr in [relaxed_points, unrelaxed_points, query_points]):
                raise ValueError("Invalid input shapes")
                
            # Create Delaunay triangulation
            try:
                tri_relaxed = Delaunay(relaxed_points)
                tri_unrelaxed = Delaunay(unrelaxed_points)
            except Exception as e:
                self.logger.error(f"Error creating Delaunay triangulation: {str(e)}")
                return None
                
            # Calculate circumcenters
            try:
                circumcenters_relaxed = np.array([
                    self._calculate_circumcenter(relaxed_points[simplex])
                    for simplex in tri_relaxed.simplices
                ])
                circumcenters_unrelaxed = np.array([
                    self._calculate_circumcenter(unrelaxed_points[simplex])
                    for simplex in tri_unrelaxed.simplices
                ])
            except Exception as e:
                self.logger.error(f"Error calculating circumcenters: {str(e)}")
                return None
                
            # Interpolate circumcenters
            try:
                rbf_interpolator = RBFInterpolator(
                    unrelaxed_points,
                    relaxed_points,
                    kernel='thin_plate_spline'
                )
                circumcenters_rbf = rbf_interpolator(circumcenters_unrelaxed)
            except Exception as e:
                self.logger.error(f"Error interpolating circumcenters: {str(e)}")
                return None
                
            # Process query points
            pristine_center = np.mean(pristine_cell, axis=0)
            pristine_cell_extended = np.vstack([pristine_cell, pristine_center])
            
            interpolated_pristine = []
            interpolated_unrelaxed = []
            
            for query_point in query_points:
                try:
                    # Find containing Voronoi cell
                    indices, nearest_idx = self._find_voronoi_cell(
                        query_point,
                        relaxed_points,
                        tri_unrelaxed,
                        circumcenters_rbf
                    )
                    
                    if indices is None or nearest_idx is None:
                        raise ValueError("Failed to find Voronoi cell")
                        
                    # Get vertices
                    vertices_relaxed = np.vstack([
                        relaxed_points[nearest_idx],
                        circumcenters_rbf[indices]
                    ])
                    vertices_unrelaxed = np.vstack([
                        unrelaxed_points[nearest_idx],
                        circumcenters_unrelaxed[indices]
                    ])
                    
                    # Interpolate point
                    rbf_voronoi = RBFInterpolator(
                        vertices_relaxed,
                        vertices_unrelaxed,
                        kernel='thin_plate_spline'
                    )
                    point = rbf_voronoi(query_point.reshape(1, -1))[0]
                    
                    # Map to cells
                    pristine_point = self._map_to_cell(
                        point.reshape(1, -1),
                        vertices_unrelaxed,
                        pristine_cell_extended,
                        unrelaxed_points[nearest_idx],
                        pristine_center
                    )
                    
                    if pristine_point is None:
                        raise ValueError("Failed to map point to pristine cell")
                        
                    unrelaxed_indices, _ = self._find_voronoi_cell(
                        unrelaxed_points[nearest_idx],
                        unrelaxed_points,
                        tri_unrelaxed,
                        circumcenters_unrelaxed
                    )
                    
                    if unrelaxed_indices is None:
                        raise ValueError("Failed to find unrelaxed Voronoi cell")
                        
                    unrelaxed_vertices = np.vstack([
                        unrelaxed_points[nearest_idx],
                        circumcenters_unrelaxed[unrelaxed_indices]
                    ])
                    
                    unrelaxed_point = self._map_to_cell(
                        point.reshape(1, -1),
                        vertices_unrelaxed,
                        unrelaxed_vertices,
                        unrelaxed_points[nearest_idx],
                        unrelaxed_points[nearest_idx]
                    )
                    
                    if unrelaxed_point is None:
                        raise ValueError("Failed to map point to unrelaxed cell")
                        
                    interpolated_pristine.append(pristine_point[0])
                    interpolated_unrelaxed.append(unrelaxed_point[0])
                    
                except Exception as e:
                    self.logger.error(f"Error processing query point {query_point}: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    continue
                    
            if not interpolated_pristine or not interpolated_unrelaxed:
                self.logger.error("No points were successfully interpolated")
                return None
                
            return (np.array(interpolated_pristine),
                    np.array(interpolated_unrelaxed),
                    circumcenters_unrelaxed,
                    circumcenters_rbf)
                    
        except Exception as e:
            self.logger.error(f"Error interpolating displacements: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def _calculate_circumcenter(self, triangle: np.ndarray) -> np.ndarray:
        """Calculate circumcenter of a triangle with error handling."""
        try:
            a, b, c = triangle
            epsilon = 1e-10
            
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(d) < epsilon:
                return np.mean(triangle, axis=0)
                
            ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) +
                  (b[0]**2 + b[1]**2) * (c[1] - a[1]) +
                  (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
            uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) +
                  (b[0]**2 + b[1]**2) * (a[0] - c[0]) +
                  (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
            
            return np.array([ux, uy])
            
        except Exception as e:
            self.logger.error(f"Error calculating circumcenter: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    def _find_voronoi_cell(
        self,
        query_point: np.ndarray,
        points: np.ndarray,
        tri: Delaunay,
        circumcenters: np.ndarray
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Find Voronoi cell containing query point with error handling."""
        try:
            distances = np.sum((points - query_point)**2, axis=1)
            nearest_point_index = np.argmin(distances)
            
            simplices_containing_point = np.where(
                (tri.simplices == nearest_point_index).any(axis=1)
            )[0]
            
            if len(simplices_containing_point) == 0:
                raise ValueError("No simplices found containing nearest point")
                
            angles = np.arctan2(
                circumcenters[simplices_containing_point][:, 1] - points[nearest_point_index, 1],
                circumcenters[simplices_containing_point][:, 0] - points[nearest_point_index, 0]
            )
            
            cell_indices = simplices_containing_point[np.argsort(angles)]
            return cell_indices, nearest_point_index
            
        except Exception as e:
            self.logger.error(f"Error finding Voronoi cell: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def _map_to_cell(
        self,
        points: np.ndarray,
        original_vertices: np.ndarray,
        target_vertices: np.ndarray,
        original_center: np.ndarray,
        target_center: np.ndarray
    ) -> Optional[np.ndarray]:
        """Map points from one Voronoi cell to another with error handling."""
        try:
            original_vertices_centered = original_vertices - (original_center - target_center)
            target_vertices_centered = target_vertices - target_center
            
            cost_matrix = cdist(original_vertices_centered, target_vertices_centered)
            _, col_ind = linear_sum_assignment(cost_matrix)
            target_vertices_matched = target_vertices_centered[col_ind]
            
            tps = RBFInterpolator(
                original_vertices_centered,
                target_vertices_matched,
                kernel='thin_plate_spline'
            )
            
            points_centered = points - (original_center - target_center)
            transformed_points = tps(points_centered)
            mapped_points = transformed_points + target_center
            
            return mapped_points
            
        except Exception as e:
            self.logger.error(f"Error mapping points to cell: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

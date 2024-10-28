from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import logging
import traceback

class DistancePlotter:
    """Class for creating publication-quality plots of distance metrics."""
    
    def __init__(self, style_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize plotter with optional style settings.
        
        Args:
            style_path: Path to matplotlib style file
            logger: Optional logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            if style_path:
                if not Path(style_path).exists():
                    raise FileNotFoundError(f"Style file not found: {style_path}")
                plt.style.use(style_path)
                
            # Default corruption factors
            self.corruption_factors = np.logspace(-4.3, -0.3, 12)
            
        except Exception as e:
            self.logger.error(f"Error initializing DistancePlotter: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def setup_figure(
        self,
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 300
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Set up figure with standard settings."""
        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            return fig, ax
        except Exception as e:
            self.logger.error(f"Error setting up figure: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def plot_distance_vs_corruption(
        self,
        distances: np.ndarray,
        output_path: str,
        title: str = "",
        xlabel: str = "Corruption Factor",
        ylabel: str = "Distance (Å)",
        with_gradients: bool = True,
        gradient_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Create scatter plot of distances vs corruption factors.
        
        Args:
            distances: Array of distance values
            output_path: Path to save plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            with_gradients: Whether to add gradient backgrounds
            gradient_thresholds: Dict of threshold values for gradients
        """
        try:
            # Validate inputs
            if distances.size == 0:
                raise ValueError("Empty distances array")
                
            # Set up figure
            fig, ax = self.setup_figure()
            
            # Create logarithmic normalization for colormap
            norm = mcolors.LogNorm(vmin=1e-5, vmax=1e0)
            
            # Plot data points
            try:
                scatter = ax.scatter(
                    self.corruption_factors,
                    distances,
                    c=self.corruption_factors,
                    cmap=plt.colormaps['RdYlGn_r'],
                    norm=norm,
                    s=100,
                    edgecolors='black',
                    alpha=0.7,
                    zorder=3
                )
            except Exception as e:
                self.logger.error(f"Error creating scatter plot: {str(e)}")
                raise
                
            if with_gradients and gradient_thresholds:
                try:
                    self._add_gradient_backgrounds(ax, gradient_thresholds)
                except Exception as e:
                    self.logger.warning(f"Failed to add gradient backgrounds: {str(e)}")
            
            # Customize plot
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Add colorbar
            try:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Model Weight Corruption Factor', fontsize=12)
                log_ticks = np.logspace(-5, 0, 6)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])
            except Exception as e:
                self.logger.warning(f"Failed to create colorbar: {str(e)}")
            
            # Save plot
            try:
                plt.savefig(output_path, bbox_inches='tight')
                self.logger.info(f"Saved plot to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save plot: {str(e)}")
                raise
            finally:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating distance vs corruption plot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    def plot_distance_comparison(
        self,
        distances1: np.ndarray,
        distances2: np.ndarray,
        output_path: str,
        labels: Tuple[str, str],
        title: str = "",
        xlabel: str = "Distance 1 (Å)",
        ylabel: str = "Distance 2 (Å)",
        with_diagonal: bool = True
    ):
        """
        Create comparison plot between two sets of distances.
        
        Args:
            distances1: First set of distances
            distances2: Second set of distances
            output_path: Path to save plot
            labels: Labels for the two datasets
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            with_diagonal: Whether to add diagonal line
        """
        try:
            # Validate inputs
            if distances1.size != distances2.size:
                raise ValueError("Distance arrays must have same size")
                
            # Set up figure
            fig, ax = self.setup_figure()
            
            # Create logarithmic normalization for colormap
            norm = mcolors.LogNorm(vmin=1e-5, vmax=1e0)
            
            # Plot data points
            try:
                scatter = ax.scatter(
                    distances1,
                    distances2,
                    c=self.corruption_factors,
                    cmap=plt.colormaps['RdYlGn_r'],
                    norm=norm,
                    s=100,
                    edgecolors='black',
                    alpha=0.7,
                    zorder=3,
                    label=labels[0]
                )
            except Exception as e:
                self.logger.error(f"Error creating scatter plot: {str(e)}")
                raise
                
            if with_diagonal:
                try:
                    # Add diagonal line
                    lims = [
                        np.min([ax.get_xlim(), ax.get_ylim()]),
                        np.max([ax.get_xlim(), ax.get_ylim()])
                    ]
                    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=1)
                except Exception as e:
                    self.logger.warning(f"Failed to add diagonal line: {str(e)}")
            
            # Customize plot
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Add colorbar
            try:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Model Weight Corruption Factor', fontsize=12)
                log_ticks = np.logspace(-5, 0, 6)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])
            except Exception as e:
                self.logger.warning(f"Failed to create colorbar: {str(e)}")
            
            # Add legend
            ax.legend()
            
            # Save plot
            try:
                plt.savefig(output_path, bbox_inches='tight')
                self.logger.info(f"Saved plot to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save plot: {str(e)}")
                raise
            finally:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating distance comparison plot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
            
    @staticmethod
    def _add_gradient_backgrounds(
        ax: plt.Axes,
        thresholds: Dict[str, float],
        alpha: float = 0.3
    ):
        """Add gradient backgrounds to highlight regions of interest."""
        try:
            for label, threshold in thresholds.items():
                gradient = np.linspace(0, alpha, 100)
                gradient = gradient.reshape(-1, 1)
                extent = [threshold * 0.9, threshold * 1.1, ax.get_ylim()[0], ax.get_ylim()[1]]
                ax.imshow(
                    gradient,
                    aspect='auto',
                    extent=extent,
                    origin='lower',
                    cmap='Greys',
                    alpha=alpha
                )
                ax.axvline(threshold, color='grey', linestyle='--', alpha=0.5)
        except Exception as e:
            raise ValueError(f"Error adding gradient backgrounds: {str(e)}")

class VoronoiPlotter:
    """Class for creating Voronoi diagram visualizations."""
    
    @staticmethod
    def plot_voronoi_diagram(
        vor: 'Voronoi',
        points: np.ndarray,
        output_path: str,
        title: str = "",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None
    ):
        """
        Create visualization of Voronoi diagram with points.
        
        Args:
            vor: Scipy Voronoi object
            points: Points to plot
            output_path: Path to save plot
            title: Plot title
            xlim: X-axis limits
            ylim: Y-axis limits
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot Voronoi diagram
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            line_colors='gray',
            line_width=1,
            line_alpha=0.6,
            point_size=2
        )
        
        # Plot points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c='blue',
            s=20,
            label='Points'
        )
        
        # Customize plot
        ax.set_aspect('equal')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_title(title)
        ax.legend()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

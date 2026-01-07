"""
Traffic Density Estimation Module
Converts vehicle counts to traffic density metrics with advanced features
"""

import numpy as np
import cv2
import yaml
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DensityEstimator:
    """Advanced traffic density estimation from vehicle detections"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize density estimator
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.density_config = self.config['density_estimation']
        self.paths = self.config['paths']
        self.num_zones = self.density_config['roi_zones']
        
        # Temporal smoothing history
        self.density_history = {}
        self.max_history_length = self.density_config['smoothing_window']
        
        # Density classification thresholds
        self.threshold_low = self.density_config['density_threshold_low']
        self.threshold_medium = self.density_config['density_threshold_medium']
        self.threshold_high = self.density_config['density_threshold_high']
        
        logger.info(f"Density estimator initialized with {self.num_zones} zones")
    
    def define_zones(
        self, 
        image_shape: Tuple[int, int],
        zone_type: str = 'horizontal'
    ) -> List[Dict]:
        """
        Define ROI zones in the image
        
        Args:
            image_shape: (height, width) of the image
            zone_type: 'horizontal' for lanes, 'grid' for grid layout, 'custom'
            
        Returns:
            zones: List of zone dictionaries with coordinates
        """
        height, width = image_shape[:2]
        zones = []
        
        if zone_type == 'horizontal':
            # Divide image into horizontal zones (lanes)
            zone_height = height // self.num_zones
            
            for i in range(self.num_zones):
                zone = {
                    'id': i,
                    'name': f'Lane_{i+1}',
                    'bbox': [
                        0,
                        i * zone_height,
                        width,
                        min((i + 1) * zone_height, height)
                    ],
                    'area': width * zone_height,
                    'type': 'lane'
                }
                zones.append(zone)
        
        elif zone_type == 'vertical':
            # Vertical zones
            zone_width = width // self.num_zones
            
            for i in range(self.num_zones):
                zone = {
                    'id': i,
                    'name': f'Section_{i+1}',
                    'bbox': [
                        i * zone_width,
                        0,
                        min((i + 1) * zone_width, width),
                        height
                    ],
                    'area': zone_width * height,
                    'type': 'section'
                }
                zones.append(zone)
        
        elif zone_type == 'grid':
            # Grid layout (2x2 for 4 zones)
            grid_rows = int(np.sqrt(self.num_zones))
            grid_cols = (self.num_zones + grid_rows - 1) // grid_rows
            
            zone_height = height // grid_rows
            zone_width = width // grid_cols
            
            zone_id = 0
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if zone_id >= self.num_zones:
                        break
                    
                    zone = {
                        'id': zone_id,
                        'name': f'Grid_{row+1}_{col+1}',
                        'bbox': [
                            col * zone_width,
                            row * zone_height,
                            min((col + 1) * zone_width, width),
                            min((row + 1) * zone_height, height)
                        ],
                        'area': zone_width * zone_height,
                        'type': 'grid'
                    }
                    zones.append(zone)
                    zone_id += 1
        
        return zones
    
    def calculate_zone_density(
        self,
        detections: List[Dict],
        zones: List[Dict],
        normalize: bool = True
    ) -> Dict[int, Dict]:
        """
        Calculate vehicle density per zone with detailed metrics
        
        Args:
            detections: List of vehicle detections
            zones: List of zone definitions
            normalize: Whether to normalize density by area
            
        Returns:
            zone_densities: Dictionary mapping zone_id to density metrics
        """
        zone_densities = {}
        
        for zone in zones:
            zone_id = zone['id']
            x1, y1, x2, y2 = zone['bbox']
            vehicles_in_zone = []
            total_vehicle_area = 0
            
            # Count vehicles whose centers are in this zone
            for det in detections:
                bx1, by1, bx2, by2 = det['bbox']
                center_x = (bx1 + bx2) / 2
                center_y = (by1 + by2) / 2
                
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    vehicles_in_zone.append(det)
                    total_vehicle_area += det.get('area', 0)
            
            vehicle_count = len(vehicles_in_zone)
            
            # Calculate density metrics
            if normalize:
                # Vehicles per unit area (normalized by 10000 pixels)
                density = (vehicle_count / zone['area']) * 10000
                
                # Area occupancy ratio (% of zone covered by vehicles)
                occupancy_ratio = (total_vehicle_area / zone['area']) * 100
            else:
                density = vehicle_count
                occupancy_ratio = 0
            
            # Classify density level
            level, color = self._classify_density(density)
            
            # Calculate average vehicle size in zone
            avg_vehicle_size = 0
            if vehicle_count > 0:
                avg_vehicle_size = total_vehicle_area / vehicle_count
            
            # Count by vehicle type
            vehicle_types = {}
            for v in vehicles_in_zone:
                v_type = v.get('class_name', 'unknown')
                vehicle_types[v_type] = vehicle_types.get(v_type, 0) + 1
            
            zone_densities[zone_id] = {
                'zone_name': zone['name'],
                'vehicle_count': vehicle_count,
                'density': density,
                'occupancy_ratio': occupancy_ratio,
                'level': level,
                'color': color,
                'vehicles': vehicles_in_zone,
                'vehicle_types': vehicle_types,
                'avg_vehicle_size': avg_vehicle_size,
                'congestion_score': self._calculate_congestion_score(
                    density, occupancy_ratio
                )
            }
        
        return zone_densities
    
    def _classify_density(self, density: float) -> Tuple[str, Tuple[int, int, int]]:
        """
        Classify density level and assign color
        
        Args:
            density: Density value
            
        Returns:
            level: Density level string
            color: BGR color tuple
        """
        if density < self.threshold_low:
            return 'low', (0, 255, 0)  # Green
        elif density < self.threshold_medium:
            return 'medium', (0, 255, 255)  # Yellow
        elif density < self.threshold_high:
            return 'high', (0, 165, 255)  # Orange
        else:
            return 'critical', (0, 0, 255)  # Red
    
    def _calculate_congestion_score(
        self, 
        density: float, 
        occupancy_ratio: float
    ) -> float:
        """
        Calculate overall congestion score (0-100)
        
        Args:
            density: Vehicle density
            occupancy_ratio: Area occupancy ratio
            
        Returns:
            score: Congestion score
        """
        # Normalize density to 0-100 scale
        density_score = min(100, (density / self.threshold_high) * 100)
        
        # Weighted average
        score = 0.6 * density_score + 0.4 * occupancy_ratio
        
        return min(100, score)
    
    def visualize_density(
        self,
        image: np.ndarray,
        zones: List[Dict],
        zone_densities: Dict[int, Dict],
        output_path: str = None,
        show_details: bool = True
    ) -> np.ndarray:
        """
        Visualize traffic density on image with enhanced details
        
        Args:
            image: Input image
            zones: Zone definitions
            zone_densities: Density metrics per zone
            output_path: Path to save visualization
            show_details: Whether to show detailed information
            
        Returns:
            vis_image: Annotated image
        """
        vis_image = image.copy()
        
        # Draw zones with color-coded density
        for zone in zones:
            zone_id = zone['id']
            x1, y1, x2, y2 = zone['bbox']
            density_info = zone_densities[zone_id]
            
            # Draw semi-transparent overlay
            overlay = vis_image.copy()
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                density_info['color'],
                -1
            )
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Draw zone border
            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                density_info['color'],
                3
            )
            
            # Add text information
            if show_details:
                texts = [
                    f"{density_info['zone_name']}",
                    f"Vehicles: {density_info['vehicle_count']}",
                    f"Level: {density_info['level'].upper()}",
                    f"Score: {density_info['congestion_score']:.1f}"
                ]
                
                y_offset = y1 + 25
                for text in texts:
                    # Text background
                    (text_w, text_h), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        vis_image,
                        (x1 + 5, y_offset - text_h - 5),
                        (x1 + text_w + 10, y_offset + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        vis_image,
                        text,
                        (x1 + 8, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    y_offset += 30
        
        # Add legend
        vis_image = self._add_legend(vis_image)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def _add_legend(self, image: np.ndarray) -> np.ndarray:
        """Add density level legend to image"""
        legend_items = [
            ('Low', (0, 255, 0)),
            ('Medium', (0, 255, 255)),
            ('High', (0, 165, 255)),
            ('Critical', (0, 0, 255))
        ]
        
        # Legend position (top-right)
        x_start = image.shape[1] - 200
        y_start = 20
        
        for i, (label, color) in enumerate(legend_items):
            y = y_start + i * 35
            
            # Color box
            cv2.rectangle(
                image,
                (x_start, y),
                (x_start + 30, y + 25),
                color,
                -1
            )
            cv2.rectangle(
                image,
                (x_start, y),
                (x_start + 30, y + 25),
                (255, 255, 255),
                2
            )
            
            # Label
            cv2.putText(
                image,
                label,
                (x_start + 40, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return image
    
    def plot_density_graph(
        self,
        zone_densities: Dict[int, Dict],
        output_path: str = None,
        plot_type: str = 'bar'
    ):
        """
        Plot comprehensive density visualization
        
        Args:
            zone_densities: Density metrics per zone
            output_path: Path to save plot
            plot_type: Type of plot ('bar', 'pie', 'comprehensive')
        """
        zones = sorted(zone_densities.keys())
        
        if plot_type == 'comprehensive':
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Vehicle count bar chart
            ax1 = fig.add_subplot(gs[0, 0])
            counts = [zone_densities[z]['vehicle_count'] for z in zones]
            colors = [zone_densities[z]['color'][::-1] for z in zones]
            colors = [(r/255, g/255, b/255) for r, g, b in colors]
            
            ax1.bar(zones, counts, color=colors, edgecolor='black', linewidth=2)
            ax1.set_xlabel('Zone ID', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Vehicle Count', fontsize=12, fontweight='bold')
            ax1.set_title('Vehicle Count per Zone', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # 2. Density metric
            ax2 = fig.add_subplot(gs[0, 1])
            densities = [zone_densities[z]['density'] for z in zones]
            
            ax2.bar(zones, densities, color=colors, edgecolor='black', linewidth=2)
            ax2.set_xlabel('Zone ID', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Density (vehicles/10k px²)', fontsize=12, fontweight='bold')
            ax2.set_title('Traffic Density per Zone', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add threshold lines
            ax2.axhline(y=self.threshold_low, color='green', linestyle='--', 
                       linewidth=2, label='Low', alpha=0.7)
            ax2.axhline(y=self.threshold_medium, color='yellow', linestyle='--', 
                       linewidth=2, label='Medium', alpha=0.7)
            ax2.axhline(y=self.threshold_high, color='orange', linestyle='--', 
                       linewidth=2, label='High', alpha=0.7)
            ax2.legend(fontsize=10)
            
            # 3. Congestion score
            ax3 = fig.add_subplot(gs[1, 0])
            scores = [zone_densities[z]['congestion_score'] for z in zones]
            
            bars = ax3.barh(zones, scores, color=colors, edgecolor='black', linewidth=2)
            ax3.set_ylabel('Zone ID', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Congestion Score (0-100)', fontsize=12, fontweight='bold')
            ax3.set_title('Congestion Score per Zone', fontsize=14, fontweight='bold')
            ax3.set_xlim(0, 100)
            ax3.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax3.text(width + 2, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}', ha='left', va='center', fontsize=10)
            
            # 4. Vehicle type distribution
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Aggregate vehicle types across all zones
            total_types = {}
            for zone_id in zones:
                for v_type, count in zone_densities[zone_id]['vehicle_types'].items():
                    total_types[v_type] = total_types.get(v_type, 0) + count
            
            if total_types:
                labels = list(total_types.keys())
                sizes = list(total_types.values())
                
                ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=plt.cm.Set3.colors[:len(labels)])
                ax4.set_title('Vehicle Type Distribution', fontsize=14, fontweight='bold')
            
            plt.suptitle('Traffic Density Analysis', fontsize=16, fontweight='bold', y=0.995)
            
        elif plot_type == 'bar':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            counts = [zone_densities[z]['vehicle_count'] for z in zones]
            densities = [zone_densities[z]['density'] for z in zones]
            colors = [zone_densities[z]['color'][::-1] for z in zones]
            colors = [(r/255, g/255, b/255) for r, g, b in colors]
            
            ax1.bar(zones, counts, color=colors, edgecolor='black')
            ax1.set_xlabel('Zone ID', fontsize=12)
            ax1.set_ylabel('Vehicle Count', fontsize=12)
            ax1.set_title('Vehicle Count per Zone', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            ax2.bar(zones, densities, color=colors, edgecolor='black')
            ax2.set_xlabel('Zone ID', fontsize=12)
            ax2.set_ylabel('Density (vehicles/10k px²)', fontsize=12)
            ax2.set_title('Traffic Density per Zone', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Density plot saved to {output_path}")
        else:
            default_path = self.paths['density_dir'] + '/density_graph.png'
            Path(default_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def temporal_smoothing(
        self,
        zone_densities: Dict[int, Dict],
        zone_id: int = None
    ) -> Dict[int, Dict]:
        """
        Apply temporal smoothing to density estimates
        
        Args:
            zone_densities: Current density measurements
            zone_id: Specific zone to smooth (None for all zones)
            
        Returns:
            smoothed_densities: Time-averaged density metrics
        """
        # Store current densities in history
        for z_id, z_data in zone_densities.items():
            if z_id not in self.density_history:
                self.density_history[z_id] = deque(maxlen=self.max_history_length)
            self.density_history[z_id].append(z_data)
        
        smoothed = {}
        
        zones_to_process = [zone_id] if zone_id is not None else zone_densities.keys()
        
        for z_id in zones_to_process:
            if z_id not in self.density_history or not self.density_history[z_id]:
                smoothed[z_id] = zone_densities[z_id]
                continue
            
            history = list(self.density_history[z_id])
            
            # Calculate moving averages
            counts = [d['vehicle_count'] for d in history]
            densities = [d['density'] for d in history]
            occupancies = [d['occupancy_ratio'] for d in history]
            scores = [d['congestion_score'] for d in history]
            
            avg_count = int(np.mean(counts))
            avg_density = np.mean(densities)
            avg_occupancy = np.mean(occupancies)
            avg_score = np.mean(scores)
            
            # Use most recent classification
            latest = history[-1]
            level, color = self._classify_density(avg_density)
            
            smoothed[z_id] = {
                'zone_name': latest['zone_name'],
                'vehicle_count': avg_count,
                'density': avg_density,
                'occupancy_ratio': avg_occupancy,
                'level': level,
                'color': color,
                'congestion_score': avg_score,
                'vehicle_types': latest['vehicle_types'],
                'avg_vehicle_size': latest['avg_vehicle_size']
            }
        
        return smoothed
    
    def compare_zone_densities(
        self,
        zone_densities: Dict[int, Dict]
    ) -> Dict[str, any]:
        """
        Compare densities across zones and provide insights
        
        Args:
            zone_densities: Density metrics per zone
            
        Returns:
            comparison: Dictionary with comparison metrics
        """
        if not zone_densities:
            return {}
        
        counts = [z['vehicle_count'] for z in zone_densities.values()]
        densities = [z['density'] for z in zone_densities.values()]
        scores = [z['congestion_score'] for z in zone_densities.values()]
        
        # Find extreme zones
        max_density_zone = max(zone_densities.items(), 
                              key=lambda x: x[1]['density'])
        min_density_zone = min(zone_densities.items(), 
                              key=lambda x: x[1]['density'])
        
        comparison = {
            'total_vehicles': sum(counts),
            'avg_density': np.mean(densities),
            'std_density': np.std(densities),
            'avg_congestion_score': np.mean(scores),
            'most_congested_zone': {
                'zone_id': max_density_zone[0],
                'zone_name': max_density_zone[1]['zone_name'],
                'density': max_density_zone[1]['density'],
                'vehicle_count': max_density_zone[1]['vehicle_count']
            },
            'least_congested_zone': {
                'zone_id': min_density_zone[0],
                'zone_name': min_density_zone[1]['zone_name'],
                'density': min_density_zone[1]['density'],
                'vehicle_count': min_density_zone[1]['vehicle_count']
            },
            'density_variance': np.var(densities),
            'congestion_levels': self._count_congestion_levels(zone_densities)
        }
        
        return comparison
    
    def _count_congestion_levels(
        self,
        zone_densities: Dict[int, Dict]
    ) -> Dict[str, int]:
        """Count zones by congestion level"""
        levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for zone_data in zone_densities.values():
            level = zone_data['level']
            levels[level] = levels.get(level, 0) + 1
        
        return levels
    
    def export_density_data(
        self,
        zone_densities: Dict[int, Dict],
        output_path: str,
        format: str = 'json'
    ):
        """
        Export density data to file
        
        Args:
            zone_densities: Density metrics
            output_path: Output file path
            format: Export format ('json', 'csv')
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert to JSON-serializable format
            export_data = {}
            for zone_id, data in zone_densities.items():
                export_data[str(zone_id)] = {
                    'zone_name': data['zone_name'],
                    'vehicle_count': int(data['vehicle_count']),
                    'density': float(data['density']),
                    'occupancy_ratio': float(data['occupancy_ratio']),
                    'level': data['level'],
                    'congestion_score': float(data['congestion_score']),
                    'vehicle_types': data['vehicle_types']
                }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Zone ID', 'Zone Name', 'Vehicle Count', 'Density',
                    'Occupancy Ratio', 'Level', 'Congestion Score'
                ])
                
                for zone_id, data in zone_densities.items():
                    writer.writerow([
                        zone_id,
                        data['zone_name'],
                        data['vehicle_count'],
                        f"{data['density']:.2f}",
                        f"{data['occupancy_ratio']:.2f}",
                        data['level'],
                        f"{data['congestion_score']:.2f}"
                    ])
        
        logger.info(f"Density data exported to {output_path}")
    
    def reset_history(self):
        """Reset temporal smoothing history"""
        self.density_history.clear()
        logger.info("Density history reset")


if __name__ == "__main__":
    # Example usage
    estimator = DensityEstimator()
    
    # Sample detections
    sample_detections = [
        {'bbox': [100, 50, 200, 150], 'confidence': 0.9, 'class': 2, 
         'class_name': 'car', 'center': (150, 100), 'area': 10000},
        {'bbox': [300, 250, 400, 350], 'confidence': 0.85, 'class': 2,
         'class_name': 'car', 'center': (350, 300), 'area': 10000},
        {'bbox': [150, 450, 250, 550], 'confidence': 0.88, 'class': 3,
         'class_name': 'motorcycle', 'center': (200, 500), 'area': 5000},
        {'bbox': [500, 100, 650, 250], 'confidence': 0.92, 'class': 7,
         'class_name': 'truck', 'center': (575, 175), 'area': 22500},
    ]
    
    # Define zones
    image_shape = (720, 1280, 3)
    zones = estimator.define_zones(image_shape, zone_type='horizontal')
    
    # Calculate densities
    densities = estimator.calculate_zone_density(sample_detections, zones)
    
    # Print results
    print("\n" + "="*60)
    print("TRAFFIC DENSITY ESTIMATION RESULTS")
    print("="*60)
    
    for zone_id, info in densities.items():
        print(f"\n{info['zone_name']}:")
        print(f"  Vehicles: {info['vehicle_count']}")
        print(f"  Density: {info['density']:.2f}")
        print(f"  Occupancy: {info['occupancy_ratio']:.2f}%")
        print(f"  Level: {info['level'].upper()}")
        print(f"  Congestion Score: {info['congestion_score']:.1f}")
        if info['vehicle_types']:
            print(f"  Types: {info['vehicle_types']}")
    
    # Comparison
    comparison = estimator.compare_zone_densities(densities)
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total Vehicles: {comparison['total_vehicles']}")
    print(f"Average Density: {comparison['avg_density']:.2f}")
    print(f"Most Congested: {comparison['most_congested_zone']['zone_name']} "
          f"({comparison['most_congested_zone']['vehicle_count']} vehicles)")
    print(f"Least Congested: {comparison['least_congested_zone']['zone_name']} "
          f"({comparison['least_congested_zone']['vehicle_count']} vehicles)")
    print("="*60 + "\n")
    
    # Generate plot
    estimator.plot_density_graph(densities, plot_type='comprehensive')
    print("Density visualization saved!")
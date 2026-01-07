"""
Visualization module for intelligent lane control decisions.
Displays lane configurations, traffic flow, and decision rationale.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class DecisionVisualizer:
    """Visualize lane control decisions and traffic patterns"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize visualizer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['output_dirs']['decisions'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lane_colors = {
            'inbound': '#FF6B6B',
            'outbound': '#4ECDC4',
            'bidirectional': '#95E1D3',
            'closed': '#CCCCCC'
        }
    
    def visualize_lane_configuration(self, current_config, predicted_config, 
                                    traffic_density, timestamp=None):
        """
        Visualize current vs predicted lane configuration
        
        Args:
            current_config: dict with current lane assignments
            predicted_config: dict with AI-recommended assignments
            traffic_density: dict with density values per direction
            timestamp: datetime object for labeling
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        if timestamp is None:
            timestamp = datetime.now()
        
        fig.suptitle(f'Lane Configuration Analysis - {timestamp.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Current Configuration
        ax1 = axes[0, 0]
        self._plot_lane_config(ax1, current_config, 'Current Configuration')
        
        # 2. Predicted Configuration
        ax2 = axes[0, 1]
        self._plot_lane_config(ax2, predicted_config, 'AI-Recommended Configuration')
        
        # 3. Traffic Density Comparison
        ax3 = axes[1, 0]
        self._plot_density_comparison(ax3, traffic_density, current_config)
        
        # 4. Efficiency Metrics
        ax4 = axes[1, 1]
        self._plot_efficiency_metrics(ax4, current_config, predicted_config, traffic_density)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f'lane_config_{timestamp.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
        return fig
    
    def _plot_lane_config(self, ax, config, title):
        """Plot lane configuration as visual diagram"""
        num_lanes = config.get('total_lanes', 4)
        lane_assignments = config.get('lanes', ['inbound'] * num_lanes)
        
        # Create lane visualization
        for i, assignment in enumerate(lane_assignments):
            color = self.lane_colors.get(assignment, '#999999')
            ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
            ax.text(0.5, i, f'Lane {i+1}\n{assignment.upper()}', 
                   ha='center', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, num_lanes - 0.5)
        ax.set_yticks(range(num_lanes))
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='black') 
                          for color in self.lane_colors.values()]
        ax.legend(legend_elements, list(self.lane_colors.keys()), 
                 loc='upper right', fontsize=9)
    
    def _plot_density_comparison(self, ax, traffic_density, config):
        """Plot traffic density by direction"""
        directions = list(traffic_density.keys())
        densities = list(traffic_density.values())
        
        # Count lanes per direction
        lane_counts = {}
        for direction in directions:
            lane_counts[direction] = sum(1 for lane in config.get('lanes', []) 
                                        if lane == direction or lane == 'bidirectional')
        
        x = np.arange(len(directions))
        width = 0.35
        
        ax.bar(x - width/2, densities, width, label='Traffic Density', 
              color='#3498db', alpha=0.7)
        ax.bar(x + width/2, [lane_counts.get(d, 0) for d in directions], 
              width, label='Allocated Lanes', color='#2ecc71', alpha=0.7)
        
        ax.set_xlabel('Direction', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Traffic Density vs Lane Allocation', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(directions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_efficiency_metrics(self, ax, current_config, predicted_config, 
                                 traffic_density):
        """Plot efficiency metrics comparison"""
        # Calculate metrics
        current_efficiency = self._calculate_efficiency(current_config, traffic_density)
        predicted_efficiency = self._calculate_efficiency(predicted_config, traffic_density)
        
        metrics = ['Overall\nEfficiency', 'Load\nBalance', 'Throughput\nPotential']
        current_scores = [current_efficiency['overall'], 
                         current_efficiency['balance'], 
                         current_efficiency['throughput']]
        predicted_scores = [predicted_efficiency['overall'], 
                           predicted_efficiency['balance'], 
                           predicted_efficiency['throughput']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, current_scores, width, label='Current', 
              color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, predicted_scores, width, label='Predicted', 
              color='#27ae60', alpha=0.7)
        
        ax.set_ylabel('Score (0-100)', fontsize=11)
        ax.set_title('Efficiency Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add improvement annotations
        for i, (curr, pred) in enumerate(zip(current_scores, predicted_scores)):
            improvement = pred - curr
            if improvement > 0:
                ax.annotate(f'+{improvement:.1f}%', 
                           xy=(i, max(curr, pred) + 3),
                           ha='center', fontsize=9, color='green', fontweight='bold')
    
    def _calculate_efficiency(self, config, traffic_density):
        """Calculate efficiency metrics for a configuration"""
        lanes = config.get('lanes', [])
        
        # Overall efficiency: how well lanes match traffic
        direction_lanes = {}
        for lane in lanes:
            direction_lanes[lane] = direction_lanes.get(lane, 0) + 1
        
        total_density = sum(traffic_density.values())
        if total_density == 0:
            overall = 50
        else:
            match_score = sum(min(direction_lanes.get(d, 0) * total_density / len(lanes), 
                                 traffic_density[d]) 
                            for d in traffic_density) / total_density
            overall = match_score * 100
        
        # Balance: how evenly distributed
        if len(direction_lanes) > 0:
            lane_counts = list(direction_lanes.values())
            balance = (1 - np.std(lane_counts) / (np.mean(lane_counts) + 1)) * 100
        else:
            balance = 50
        
        # Throughput potential
        throughput = min(100, (len(lanes) / 6) * 100)  # Assuming 6 lanes is optimal
        
        return {
            'overall': max(0, min(100, overall)),
            'balance': max(0, min(100, balance)),
            'throughput': max(0, min(100, throughput))
        }
    
    def create_interactive_timeline(self, decisions_history, save_path=None):
        """
        Create interactive Plotly timeline of decisions
        
        Args:
            decisions_history: list of dicts with timestamp, config, and metrics
            save_path: optional path to save HTML
        """
        if not decisions_history:
            print("No decision history to visualize")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(decisions_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Lane Configuration Over Time', 
                          'Traffic Density', 
                          'Efficiency Metrics'),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Lane configuration timeline
        for direction in ['inbound', 'outbound', 'bidirectional']:
            lane_counts = [sum(1 for lane in d.get('lanes', []) if lane == direction) 
                          for d in df['config']]
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=lane_counts, 
                          mode='lines+markers', name=direction.capitalize(),
                          line=dict(width=2), marker=dict(size=6)),
                row=1, col=1
            )
        
        # 2. Traffic density
        if 'traffic_density' in df.columns:
            for direction in ['inbound', 'outbound']:
                densities = [d.get(direction, 0) for d in df['traffic_density']]
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=densities,
                              mode='lines', name=f'{direction.capitalize()} Traffic',
                              line=dict(dash='dash')),
                    row=2, col=1
                )
        
        # 3. Efficiency metrics
        if 'efficiency' in df.columns:
            for metric in ['overall', 'balance', 'throughput']:
                values = [e.get(metric, 0) for e in df['efficiency']]
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=values,
                              mode='lines', name=metric.capitalize(),
                              line=dict(width=2)),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Lane Count", row=1, col=1)
        fig.update_yaxes(title_text="Density (vehicles/min)", row=2, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=3, col=1)
        
        fig.update_layout(
            height=900,
            title_text="Intelligent Lane Control Timeline",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Save if path provided
        if save_path is None:
            save_path = self.output_dir / f'timeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        fig.write_html(str(save_path))
        print(f"Saved interactive timeline to {save_path}")
        
        return fig
    
    def generate_decision_report(self, decision_data, save_path=None):
        """
        Generate comprehensive decision report
        
        Args:
            decision_data: dict containing decision details
            save_path: optional path to save report
        """
        if save_path is None:
            save_path = self.output_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("INTELLIGENT LANE CONTROL DECISION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Timestamp: {decision_data.get('timestamp', datetime.now())}\n")
            f.write(f"Decision Type: {decision_data.get('decision_type', 'AI-Based')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("CURRENT SITUATION\n")
            f.write("-" * 70 + "\n")
            traffic = decision_data.get('traffic_density', {})
            for direction, density in traffic.items():
                f.write(f"  {direction.capitalize()}: {density:.2f} vehicles/min\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("LANE CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            config = decision_data.get('config', {})
            lanes = config.get('lanes', [])
            for i, lane in enumerate(lanes):
                f.write(f"  Lane {i+1}: {lane.upper()}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("RATIONALE\n")
            f.write("-" * 70 + "\n")
            rationale = decision_data.get('rationale', 'No rationale provided')
            f.write(f"  {rationale}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("EXPECTED IMPACT\n")
            f.write("-" * 70 + "\n")
            impact = decision_data.get('expected_impact', {})
            for metric, value in impact.items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Saved decision report to {save_path}")
        return save_path


def main():
    """Demo visualization"""
    visualizer = DecisionVisualizer()
    
    # Sample data
    current = {
        'total_lanes': 4,
        'lanes': ['inbound', 'inbound', 'outbound', 'outbound']
    }
    
    predicted = {
        'total_lanes': 4,
        'lanes': ['inbound', 'inbound', 'inbound', 'outbound']
    }
    
    traffic = {
        'inbound': 85.5,
        'outbound': 32.3
    }
    
    # Visualize
    visualizer.visualize_lane_configuration(current, predicted, traffic)
    
    # Generate report
    decision_data = {
        'timestamp': datetime.now(),
        'decision_type': 'AI-Based',
        'traffic_density': traffic,
        'config': predicted,
        'rationale': 'High inbound traffic detected. Recommend allocating 3 lanes inbound, 1 outbound.',
        'expected_impact': {
            'congestion_reduction': '25%',
            'avg_travel_time': '-8 minutes',
            'throughput_increase': '18%'
        }
    }
    
    visualizer.generate_decision_report(decision_data)
    
    plt.show()


if __name__ == '__main__':
    main()
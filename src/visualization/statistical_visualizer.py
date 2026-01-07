"""
Statistical Visualizer Module
Generates comprehensive visual statistical outputs for traffic analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class StatisticalVisualizer:
    """Generate statistical visualizations for traffic management system"""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'inbound': '#FF6B6B',
            'outbound': '#4ECDC4',
            'balanced': '#95E1D3',
            'priority': '#F38181'
        }
    
    def visualize_24hour_simulation(self, simulation_data: List[Dict], 
                                   save_path: Optional[str] = None):
        """
        Create comprehensive visualization for 24-hour simulation
        
        Args:
            simulation_data: List of hourly traffic data
            save_path: Path to save figure
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"24hour_simulation_{timestamp}.png"
        
        # Create subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Prepare data
        hours = [d['hour'] for d in simulation_data]
        inbound_traffic = [d['inbound_density'] for d in simulation_data]
        outbound_traffic = [d['outbound_density'] for d in simulation_data]
        inbound_lanes = [d['inbound_lanes'] for d in simulation_data]
        outbound_lanes = [d['outbound_lanes'] for d in simulation_data]
        config_names = [d['config_name'] for d in simulation_data]
        
        # 1. Traffic Density Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(hours, inbound_traffic, 'o-', color=self.colors['inbound'], 
                linewidth=2, markersize=6, label='Inbound Traffic')
        ax1.plot(hours, outbound_traffic, 's-', color=self.colors['outbound'], 
                linewidth=2, markersize=6, label='Outbound Traffic')
        ax1.fill_between(hours, inbound_traffic, alpha=0.3, color=self.colors['inbound'])
        ax1.fill_between(hours, outbound_traffic, alpha=0.3, color=self.colors['outbound'])
        ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Traffic Density', fontsize=12, fontweight='bold')
        ax1.set_title('24-Hour Traffic Density Pattern', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Highlight rush hours
        ax1.axvspan(7, 9, alpha=0.2, color='yellow', label='Morning Rush')
        ax1.axvspan(17, 19, alpha=0.2, color='orange', label='Evening Rush')
        
        # 2. Lane Allocation Over Time
        ax2 = fig.add_subplot(gs[0, 2])
        x_pos = np.arange(len(hours))
        ax2.bar(x_pos, inbound_lanes, width=0.8, color=self.colors['inbound'], 
               alpha=0.7, label='Inbound Lanes')
        ax2.bar(x_pos, outbound_lanes, width=0.8, bottom=inbound_lanes,
               color=self.colors['outbound'], alpha=0.7, label='Outbound Lanes')
        ax2.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Lane Count', fontsize=12, fontweight='bold')
        ax2.set_title('Lane Allocation', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_xticks(range(0, 24, 4))
        ax2.set_xticklabels(range(0, 24, 4))
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 3. Traffic Ratio Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        traffic_ratio = [i/(i+o) if (i+o) > 0 else 0.5 
                        for i, o in zip(inbound_traffic, outbound_traffic)]
        scatter = ax3.scatter(hours, traffic_ratio, c=traffic_ratio, 
                            cmap='RdYlGn_r', s=100, edgecolors='black', linewidth=1.5)
        ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Balanced (0.5)')
        ax3.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Inbound Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Traffic Direction Ratio', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Inbound Ratio')
        
        # 4. Configuration Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        config_counts = pd.Series(config_names).value_counts()
        colors_pie = [self.colors.get(config.split('_')[0], '#999999') 
                     for config in config_counts.index]
        wedges, texts, autotexts = ax4.pie(config_counts.values, labels=config_counts.index,
                                           autopct='%1.1f%%', colors=colors_pie,
                                           startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Configuration Distribution', fontsize=14, fontweight='bold')
        
        # 5. Lane Efficiency Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        efficiency = []
        for i, o, il, ol in zip(inbound_traffic, outbound_traffic, 
                                inbound_lanes, outbound_lanes):
            total_traffic = i + o
            total_lanes = il + ol
            if total_lanes > 0:
                eff = (total_traffic / total_lanes) * 100
                efficiency.append(min(eff, 100))
            else:
                efficiency.append(0)
        
        ax5.bar(hours, efficiency, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
        ax5.set_title('Lane Utilization Efficiency', fontsize=14, fontweight='bold')
        ax5.axhline(y=70, color='green', linestyle='--', label='Optimal (70%)')
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Rush Hour Analysis
        ax6 = fig.add_subplot(gs[2, 0])
        rush_hours = {
            'Morning\nRush\n(7-9)': np.mean(inbound_traffic[7:10]),
            'Normal\nHours': np.mean([inbound_traffic[i] for i in range(24) 
                                     if i not in range(7, 10) and i not in range(17, 20)]),
            'Evening\nRush\n(17-19)': np.mean(outbound_traffic[17:20])
        }
        bars = ax6.bar(rush_hours.keys(), rush_hours.values(), 
                      color=['#FF6B6B', '#95E1D3', '#4ECDC4'], alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Avg Traffic Density', fontsize=12, fontweight='bold')
        ax6.set_title('Rush Hour vs Normal Traffic', fontsize=14, fontweight='bold')
        ax6.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 7. Decision Timeline
        ax7 = fig.add_subplot(gs[2, 1:])
        for i, (hour, config) in enumerate(zip(hours, config_names)):
            if 'inbound' in config:
                color = self.colors['inbound']
            elif 'outbound' in config:
                color = self.colors['outbound']
            else:
                color = self.colors['balanced']
            
            ax7.barh(i, 1, left=hour, height=0.8, color=color, alpha=0.7, edgecolor='black')
        
        ax7.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Decision Sequence', fontsize=12, fontweight='bold')
        ax7.set_title('Decision Timeline', fontsize=14, fontweight='bold')
        ax7.set_xlim(0, 24)
        ax7.set_xticks(range(0, 24, 2))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['inbound'], label='Inbound Priority'),
            Patch(facecolor=self.colors['outbound'], label='Outbound Priority'),
            Patch(facecolor=self.colors['balanced'], label='Balanced')
        ]
        ax7.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Main title
        fig.suptitle('24-Hour Traffic Management System Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Statistical visualization saved to: {save_path}")
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(self, simulation_data: List[Dict],
                                    save_path: Optional[str] = None):
        """
        Create interactive Plotly dashboard
        
        Args:
            simulation_data: List of hourly traffic data
            save_path: Path to save HTML file
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"interactive_dashboard_{timestamp}.html"
        
        # Prepare data
        df = pd.DataFrame(simulation_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Traffic Density Over Time',
                'Lane Allocation Pattern',
                'Traffic Direction Ratio',
                'Configuration Distribution',
                'Lane Efficiency Analysis',
                'Peak Hour Comparison'
            ),
            specs=[
                [{"secondary_y": False}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Traffic Density Over Time
        fig.add_trace(
            go.Scatter(x=df['hour'], y=df['inbound_density'],
                      mode='lines+markers', name='Inbound Traffic',
                      line=dict(color=self.colors['inbound'], width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['hour'], y=df['outbound_density'],
                      mode='lines+markers', name='Outbound Traffic',
                      line=dict(color=self.colors['outbound'], width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # 2. Lane Allocation
        fig.add_trace(
            go.Bar(x=df['hour'], y=df['inbound_lanes'],
                  name='Inbound Lanes', marker_color=self.colors['inbound']),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=df['hour'], y=df['outbound_lanes'],
                  name='Outbound Lanes', marker_color=self.colors['outbound']),
            row=1, col=2
        )
        
        # 3. Traffic Ratio
        df['traffic_ratio'] = df.apply(
            lambda row: row['inbound_density'] / (row['inbound_density'] + row['outbound_density'])
            if (row['inbound_density'] + row['outbound_density']) > 0 else 0.5,
            axis=1
        )
        fig.add_trace(
            go.Scatter(x=df['hour'], y=df['traffic_ratio'],
                      mode='markers', name='Traffic Ratio',
                      marker=dict(size=12, color=df['traffic_ratio'],
                                colorscale='RdYlGn_r', showscale=True,
                                colorbar=dict(x=0.46, len=0.3))),
            row=2, col=1
        )
        
        # 4. Configuration Distribution
        config_counts = df['config_name'].value_counts()
        fig.add_trace(
            go.Pie(labels=config_counts.index, values=config_counts.values,
                  marker=dict(colors=[self.colors.get(c.split('_')[0], '#999999') 
                                     for c in config_counts.index])),
            row=2, col=2
        )
        
        # 5. Lane Efficiency
        df['efficiency'] = df.apply(
            lambda row: min((row['inbound_density'] + row['outbound_density']) / 
                          (row['inbound_lanes'] + row['outbound_lanes']) * 100, 100)
            if (row['inbound_lanes'] + row['outbound_lanes']) > 0 else 0,
            axis=1
        )
        fig.add_trace(
            go.Bar(x=df['hour'], y=df['efficiency'],
                  name='Efficiency', marker_color='steelblue'),
            row=3, col=1
        )
        
        # 6. Peak Hour Comparison
        morning_rush = df[(df['hour'] >= 7) & (df['hour'] <= 9)]['inbound_density'].mean()
        evening_rush = df[(df['hour'] >= 17) & (df['hour'] <= 19)]['outbound_density'].mean()
        normal_hours = df[~df['hour'].isin(range(7, 10)) & 
                         ~df['hour'].isin(range(17, 20))]['inbound_density'].mean()
        
        fig.add_trace(
            go.Bar(x=['Morning Rush', 'Normal Hours', 'Evening Rush'],
                  y=[morning_rush, normal_hours, evening_rush],
                  marker_color=[self.colors['inbound'], 
                              self.colors['balanced'],
                              self.colors['outbound']]),
            row=3, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_xaxes(title_text="Hour", row=1, col=2)
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=3, col=1)
        
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Lanes", row=1, col=2)
        fig.update_yaxes(title_text="Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=3, col=1)
        fig.update_yaxes(title_text="Avg Density", row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Interactive Traffic Management Dashboard",
            title_font_size=20,
            hovermode='x unified'
        )
        
        # Save
        fig.write_html(str(save_path))
        print(f"✓ Interactive dashboard saved to: {save_path}")
        
        return str(save_path)
    
    def generate_summary_statistics(self, simulation_data: List[Dict],
                                   save_path: Optional[str] = None):
        """
        Generate summary statistics report
        
        Args:
            simulation_data: List of hourly traffic data
            save_path: Path to save report
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"summary_statistics_{timestamp}.txt"
        
        df = pd.DataFrame(simulation_data)
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("24-HOUR TRAFFIC SIMULATION - SUMMARY STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            # Overall Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Hours Simulated: {len(df)}\n")
            f.write(f"Average Inbound Density: {df['inbound_density'].mean():.3f}\n")
            f.write(f"Average Outbound Density: {df['outbound_density'].mean():.3f}\n")
            f.write(f"Peak Inbound Hour: {df.loc[df['inbound_density'].idxmax(), 'hour']:02d}:00 "
                   f"({df['inbound_density'].max():.3f})\n")
            f.write(f"Peak Outbound Hour: {df.loc[df['outbound_density'].idxmax(), 'hour']:02d}:00 "
                   f"({df['outbound_density'].max():.3f})\n\n")
            
            # Configuration Statistics
            f.write("CONFIGURATION STATISTICS\n")
            f.write("-"*80 + "\n")
            config_counts = df['config_name'].value_counts()
            for config, count in config_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"  {config:25s}: {count:2d} hours ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Lane Allocation Statistics
            f.write("LANE ALLOCATION STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Inbound Lanes: {df['inbound_lanes'].mean():.2f}\n")
            f.write(f"Average Outbound Lanes: {df['outbound_lanes'].mean():.2f}\n")
            f.write(f"Most Common: {df['inbound_lanes'].mode()[0]} inbound, "
                   f"{df['outbound_lanes'].mode()[0]} outbound\n\n")
            
            # Rush Hour Analysis
            f.write("RUSH HOUR ANALYSIS\n")
            f.write("-"*80 + "\n")
            morning_rush = df[(df['hour'] >= 7) & (df['hour'] <= 9)]
            evening_rush = df[(df['hour'] >= 17) & (df['hour'] <= 19)]
            
            f.write("Morning Rush (07:00-09:00):\n")
            f.write(f"  Avg Inbound Density: {morning_rush['inbound_density'].mean():.3f}\n")
            f.write(f"  Avg Outbound Density: {morning_rush['outbound_density'].mean():.3f}\n")
            f.write(f"  Most Common Config: {morning_rush['config_name'].mode()[0]}\n\n")
            
            f.write("Evening Rush (17:00-19:00):\n")
            f.write(f"  Avg Inbound Density: {evening_rush['inbound_density'].mean():.3f}\n")
            f.write(f"  Avg Outbound Density: {evening_rush['outbound_density'].mean():.3f}\n")
            f.write(f"  Most Common Config: {evening_rush['config_name'].mode()[0]}\n\n")
            
            # Efficiency Analysis
            df['efficiency'] = df.apply(
                lambda row: min((row['inbound_density'] + row['outbound_density']) / 
                              (row['inbound_lanes'] + row['outbound_lanes']) * 100, 100)
                if (row['inbound_lanes'] + row['outbound_lanes']) > 0 else 0,
                axis=1
            )
            
            f.write("EFFICIENCY ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Efficiency: {df['efficiency'].mean():.2f}%\n")
            f.write(f"Best Hour: {df.loc[df['efficiency'].idxmax(), 'hour']:02d}:00 "
                   f"({df['efficiency'].max():.2f}%)\n")
            f.write(f"Worst Hour: {df.loc[df['efficiency'].idxmin(), 'hour']:02d}:00 "
                   f"({df['efficiency'].min():.2f}%)\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Summary statistics saved to: {save_path}")
        return str(save_path)


def main():
    """Demo function"""
    # Generate sample data
    simulation_data = []
    for hour in range(24):
        if 7 <= hour <= 9:
            inbound = 0.8 + np.random.rand() * 0.1
            outbound = 0.3 + np.random.rand() * 0.1
            config = 'inbound_priority'
            inbound_lanes, outbound_lanes = 3, 1
        elif 17 <= hour <= 19:
            inbound = 0.3 + np.random.rand() * 0.1
            outbound = 0.8 + np.random.rand() * 0.1
            config = 'outbound_priority'
            inbound_lanes, outbound_lanes = 1, 3
        else:
            inbound = 0.4 + np.random.rand() * 0.2
            outbound = 0.4 + np.random.rand() * 0.2
            config = 'balanced'
            inbound_lanes, outbound_lanes = 2, 2
        
        simulation_data.append({
            'hour': hour,
            'inbound_density': inbound,
            'outbound_density': outbound,
            'config_name': config,
            'inbound_lanes': inbound_lanes,
            'outbound_lanes': outbound_lanes
        })
    
    # Create visualizer
    visualizer = StatisticalVisualizer()
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    visualizer.visualize_24hour_simulation(simulation_data)
    visualizer.create_interactive_dashboard(simulation_data)
    visualizer.generate_summary_statistics(simulation_data)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    main()
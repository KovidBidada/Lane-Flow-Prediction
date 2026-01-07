"""
Evaluation module for traffic prediction models.
Computes metrics, visualizes predictions, and generates performance reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelEvaluator :
    """Comprehensive evaluation for traffic prediction models"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['paths']['predictions_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
    
    def calculate_metrics(self, y_true, y_pred, prefix=''):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Ground truth values (n_samples, n_features)
            y_pred: Predicted values (n_samples, n_features)
            prefix: Prefix for metric names (e.g., 'train_', 'test_')
        
        Returns:
            dict: Dictionary of computed metrics
        """
        # Flatten arrays if multi-dimensional
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Core regression metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true_flat != 0
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        # Correlation
        correlation, p_value = pearsonr(y_true_flat, y_pred_flat)
        
        # Error statistics
        errors = y_pred_flat - y_true_flat
        error_std = np.std(errors)
        error_median = np.median(np.abs(errors))
        
        # Directional accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.diff(y_true.mean(axis=1) if y_true.ndim > 1 else y_true) > 0
            pred_direction = np.diff(y_pred.mean(axis=1) if y_pred.ndim > 1 else y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = None
        
        metrics = {
            f'{prefix}mse': mse,
            f'{prefix}rmse': rmse,
            f'{prefix}mae': mae,
            f'{prefix}mape': mape,
            f'{prefix}r2': r2,
            f'{prefix}correlation': correlation,
            f'{prefix}correlation_pvalue': p_value,
            f'{prefix}error_std': error_std,
            f'{prefix}error_median': error_median,
        }
        
        if directional_accuracy is not None:
            metrics[f'{prefix}directional_accuracy'] = directional_accuracy
        
        return metrics
    
    def plot_predictions_vs_actual(self, y_true, y_pred, timestamps=None, 
                                   sensor_names=None, save_path=None):
        """
        Plot predicted vs actual values
        
        Args:
            y_true: Ground truth (n_samples, n_sensors)
            y_pred: Predictions (n_samples, n_sensors)
            timestamps: Time indices for x-axis
            sensor_names: Names of sensors/locations
            save_path: Path to save figure
        """
        n_samples, n_sensors = y_true.shape
        
        # Select subset of sensors if too many
        max_sensors = 6
        if n_sensors > max_sensors:
            sensor_indices = np.linspace(0, n_sensors-1, max_sensors, dtype=int)
        else:
            sensor_indices = range(n_sensors)
        
        n_plots = len(sensor_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        if timestamps is None:
            timestamps = np.arange(n_samples)
        
        for idx, sensor_idx in enumerate(sensor_indices):
            ax = axes[idx]
            
            ax.plot(timestamps, y_true[:, sensor_idx], 
                   label='Actual', color='#2C3E50', linewidth=2, alpha=0.8)
            ax.plot(timestamps, y_pred[:, sensor_idx], 
                   label='Predicted', color='#E74C3C', linewidth=2, alpha=0.8, linestyle='--')
            
            # Add shaded error region
            errors = np.abs(y_true[:, sensor_idx] - y_pred[:, sensor_idx])
            ax.fill_between(timestamps, 
                           y_pred[:, sensor_idx] - errors, 
                           y_pred[:, sensor_idx] + errors,
                           alpha=0.2, color='#E74C3C')
            
            sensor_name = sensor_names[sensor_idx] if sensor_names else f'Sensor {sensor_idx}'
            ax.set_title(f'{sensor_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Traffic Flow', fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
        
        axes[-1].set_xlabel('Time Step', fontsize=11)
        
        plt.suptitle('Traffic Flow: Predictions vs Actual', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")
        
        return fig
    
    def plot_error_distribution(self, y_true, y_pred, save_path=None):
        """
        Plot error distribution and analysis
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_path: Path to save figure
        """
        errors = (y_pred - y_true).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Error histogram
        ax1 = axes[0, 0]
        ax1.hist(errors, bins=50, color='#3498DB', alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Prediction Error', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Q-Q plot
        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Residual plot
        ax3 = axes[1, 0]
        y_true_flat = y_true.flatten()
        ax3.scatter(y_true_flat, errors, alpha=0.3, s=10, color='#E74C3C')
        ax3.axhline(0, color='black', linestyle='--', linewidth=2)
        ax3.set_xlabel('Actual Values', fontsize=11)
        ax3.set_ylabel('Residuals', fontsize=11)
        ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Error statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        ERROR STATISTICS
        {'='*40}
        
        Mean Error:          {np.mean(errors):>10.3f}
        Std Dev:             {np.std(errors):>10.3f}
        
        Min Error:           {np.min(errors):>10.3f}
        Max Error:           {np.max(errors):>10.3f}
        
        Median Abs Error:    {np.median(np.abs(errors)):>10.3f}
        90th Percentile:     {np.percentile(np.abs(errors), 90):>10.3f}
        95th Percentile:     {np.percentile(np.abs(errors), 95):>10.3f}
        
        {'='*40}
        Error within ±5%:    {np.sum(np.abs(errors/y_true_flat) < 0.05)/len(errors)*100:>9.1f}%
        Error within ±10%:   {np.sum(np.abs(errors/y_true_flat) < 0.10)/len(errors)*100:>9.1f}%
        Error within ±20%:   {np.sum(np.abs(errors/y_true_flat) < 0.20)/len(errors)*100:>9.1f}%
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'error_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Compare metrics across different models or time periods
        
        Args:
            metrics_dict: dict mapping model names to metric dicts
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        models = list(metrics_dict.keys())
        
        # 1. Error metrics
        ax1 = axes[0]
        metrics_to_plot = ['rmse', 'mae', 'mape']
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            ax1.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)
        
        ax1.set_xlabel('Model', fontsize=11)
        ax1.set_ylabel('Error Value', fontsize=11)
        ax1.set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. R² and Correlation
        ax2 = axes[1]
        r2_values = [metrics_dict[m].get('r2', 0) for m in models]
        corr_values = [metrics_dict[m].get('correlation', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax2.bar(x - width/2, r2_values, width, label='R²', color='#27AE60', alpha=0.8)
        ax2.bar(x + width/2, corr_values, width, label='Correlation', color='#2980B9', alpha=0.8)
        
        ax2.set_xlabel('Model', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Goodness of Fit', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Directional Accuracy (if available)
        ax3 = axes[2]
        if all('directional_accuracy' in metrics_dict[m] for m in models):
            dir_acc = [metrics_dict[m]['directional_accuracy'] for m in models]
            colors = ['#E74C3C' if acc < 70 else '#F39C12' if acc < 85 else '#27AE60' 
                     for acc in dir_acc]
            
            bars = ax3.bar(models, dir_acc, color=colors, alpha=0.8, edgecolor='black')
            ax3.axhline(50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
            ax3.set_ylabel('Accuracy (%)', fontsize=11)
            ax3.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 100)
            ax3.legend()
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Directional Accuracy\nNot Available', 
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'metrics_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
        
        return fig
    
    def create_interactive_evaluation(self, y_true, y_pred, timestamps=None, 
                                     sensor_names=None, save_path=None):
        """
        Create interactive Plotly visualization of predictions
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            timestamps: Time indices
            sensor_names: Sensor names
            save_path: Path to save HTML
        """
        n_samples, n_sensors = y_true.shape
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='5T')
        
        if sensor_names is None:
            sensor_names = [f'Sensor_{i}' for i in range(n_sensors)]
        
        # Select subset for visualization
        max_sensors = 4
        if n_sensors > max_sensors:
            sensor_indices = np.linspace(0, n_sensors-1, max_sensors, dtype=int)
        else:
            sensor_indices = range(n_sensors)
        
        # Create subplots
        fig = make_subplots(
            rows=len(sensor_indices), cols=1,
            subplot_titles=[sensor_names[i] for i in sensor_indices],
            vertical_spacing=0.08
        )
        
        for idx, sensor_idx in enumerate(sensor_indices, 1):
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_true[:, sensor_idx],
                    mode='lines',
                    name=f'Actual - {sensor_names[sensor_idx]}',
                    line=dict(color='#2C3E50', width=2),
                    showlegend=(idx == 1)
                ),
                row=idx, col=1
            )
            
            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_pred[:, sensor_idx],
                    mode='lines',
                    name=f'Predicted - {sensor_names[sensor_idx]}',
                    line=dict(color='#E74C3C', width=2, dash='dash'),
                    showlegend=(idx == 1)
                ),
                row=idx, col=1
            )
            
            # Error band
            errors = np.abs(y_true[:, sensor_idx] - y_pred[:, sensor_idx])
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_pred[:, sensor_idx] + errors,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_pred[:, sensor_idx] - errors,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    fill='tonexty',
                    showlegend=(idx == 1),
                    name='Error Range' if idx == 1 else None
                ),
                row=idx, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=len(sensor_indices), col=1)
        
        for idx in range(1, len(sensor_indices) + 1):
            fig.update_yaxes(title_text="Traffic Flow", row=idx, col=1)
        
        fig.update_layout(
            height=300 * len(sensor_indices),
            title_text="Traffic Flow Predictions - Interactive View",
            hovermode='x unified',
            showlegend=True
        )
        
        if save_path is None:
            save_path = self.output_dir / f'interactive_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        fig.write_html(str(save_path))
        print(f"Saved interactive evaluation to {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, y_true, y_pred, model_name='Model', 
                                   save_path=None):
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save report
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        if save_path is None:
            save_path = self.output_dir / f'report_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAFFIC PREDICTION EVALUATION REPORT - {model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {len(y_true)}\n")
            f.write(f"Features: {y_true.shape[1] if y_true.ndim > 1 else 1}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Primary Metrics:\n")
            f.write(f"  Root Mean Squared Error (RMSE):        {metrics['rmse']:>12.4f}\n")
            f.write(f"  Mean Absolute Error (MAE):             {metrics['mae']:>12.4f}\n")
            f.write(f"  Mean Absolute Percentage Error (MAPE): {metrics['mape']:>12.2f}%\n")
            f.write(f"  R² Score:                              {metrics['r2']:>12.4f}\n\n")
            
            f.write("Statistical Measures:\n")
            f.write(f"  Correlation Coefficient:               {metrics['correlation']:>12.4f}\n")
            f.write(f"  Error Standard Deviation:              {metrics['error_std']:>12.4f}\n")
            f.write(f"  Median Absolute Error:                 {metrics['error_median']:>12.4f}\n\n")
            
            if 'directional_accuracy' in metrics:
                f.write(f"Directional Accuracy:                    {metrics['directional_accuracy']:>12.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n\n")
            
            # Provide interpretation
            if metrics['mape'] < 10:
                f.write("✓ Excellent prediction accuracy (MAPE < 10%)\n")
            elif metrics['mape'] < 20:
                f.write("✓ Good prediction accuracy (MAPE < 20%)\n")
            elif metrics['mape'] < 30:
                f.write("⚠ Moderate prediction accuracy (MAPE < 30%)\n")
            else:
                f.write("✗ Poor prediction accuracy (MAPE ≥ 30%)\n")
            
            if metrics['r2'] > 0.9:
                f.write("✓ Excellent model fit (R² > 0.9)\n")
            elif metrics['r2'] > 0.7:
                f.write("✓ Good model fit (R² > 0.7)\n")
            elif metrics['r2'] > 0.5:
                f.write("⚠ Moderate model fit (R² > 0.5)\n")
            else:
                f.write("✗ Poor model fit (R² ≤ 0.5)\n")
            
            if 'directional_accuracy' in metrics:
                if metrics['directional_accuracy'] > 85:
                    f.write("✓ Excellent directional prediction (> 85%)\n")
                elif metrics['directional_accuracy'] > 70:
                    f.write("✓ Good directional prediction (> 70%)\n")
                else:
                    f.write("⚠ Needs improvement in directional prediction\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Saved evaluation report to {save_path}")
        
        # Save metrics as JSON
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to {json_path}")
        
        return save_path
    
    def plot_horizon_performance(self, y_true, y_pred, horizon_steps, save_path=None):
        """
        Evaluate performance at different prediction horizons
        
        Args:
            y_true: Ground truth (n_samples, n_horizons, n_sensors)
            y_pred: Predictions (n_samples, n_horizons, n_sensors)
            horizon_steps: List of horizon step values
            save_path: Path to save figure
        """
        n_horizons = len(horizon_steps)
        
        mae_per_horizon = []
        rmse_per_horizon = []
        mape_per_horizon = []
        
        for h in range(n_horizons):
            metrics = self.calculate_metrics(y_true[:, h, :], y_pred[:, h, :])
            mae_per_horizon.append(metrics['mae'])
            rmse_per_horizon.append(metrics['rmse'])
            mape_per_horizon.append(metrics['mape'])
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # MAE
        axes[0].plot(horizon_steps, mae_per_horizon, marker='o', linewidth=2, 
                    markersize=8, color='#3498DB')
        axes[0].set_xlabel('Prediction Horizon', fontsize=11)
        axes[0].set_ylabel('MAE', fontsize=11)
        axes[0].set_title('Mean Absolute Error by Horizon', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # RMSE
        axes[1].plot(horizon_steps, rmse_per_horizon, marker='s', linewidth=2, 
                    markersize=8, color='#E74C3C')
        axes[1].set_xlabel('Prediction Horizon', fontsize=11)
        axes[1].set_ylabel('RMSE', fontsize=11)
        axes[1].set_title('Root Mean Squared Error by Horizon', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # MAPE
        axes[2].plot(horizon_steps, mape_per_horizon, marker='^', linewidth=2, 
                    markersize=8, color='#27AE60')
        axes[2].set_xlabel('Prediction Horizon', fontsize=11)
        axes[2].set_ylabel('MAPE (%)', fontsize=11)
        axes[2].set_title('Mean Absolute Percentage Error by Horizon', 
                         fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'horizon_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved horizon performance plot to {save_path}")
        
        return fig
    
    def save_metrics_history(self, metrics, metadata=None):
        """
        Save metrics to history for tracking over time
        
        Args:
            metrics: Dictionary of metrics
            metadata: Additional metadata (e.g., model version, parameters)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(entry)
        
        # Save to file
        history_path = self.output_dir / 'metrics_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Saved metrics history to {history_path}")


def main():
    """Demo evaluation"""
    evaluator = ModelEvaluator()
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_sensors = 100, 3
    
    # Simulate predictions with some error
    y_true = np.random.rand(n_samples, n_sensors) * 100
    y_pred = y_true + np.random.randn(n_samples, n_sensors) * 10
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate visualizations
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='5T')
    sensor_names = ['Sensor_A', 'Sensor_B', 'Sensor_C']
    
    evaluator.plot_predictions_vs_actual(y_true, y_pred, timestamps, sensor_names)
    evaluator.plot_error_distribution(y_true, y_pred)
    
    # Generate report
    evaluator.generate_evaluation_report(y_true, y_pred, model_name='DemoModel')
    
    # Create interactive plot
    evaluator.create_interactive_evaluation(y_true, y_pred, timestamps, sensor_names)
    
    # Compare multiple models
    metrics_dict = {
        'LSTM': metrics,
        'GCN': {k: v * 0.9 for k, v in metrics.items()},  # Simulated better model
        'Baseline': {k: v * 1.3 for k, v in metrics.items()}  # Simulated worse model
    }
    evaluator.plot_metrics_comparison(metrics_dict)
    
    plt.show()


if __name__ == '__main__':
    main()
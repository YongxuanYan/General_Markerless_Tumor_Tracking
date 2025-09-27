import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
from pathlib import Path

# Set academic color scheme
colors = ['#264653', '#2a9d8f', '#e9c46a', '#98fb98', '#e76f51']
sns.set_palette(sns.color_palette(colors))
sns.set_style("whitegrid")

# Increase font sizes for academic publication
plt.rcParams['font.size'] = 25
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 30


def create_standalone_legend(save_path=None):
    """
    Create standalone legend file for publication formatting.
    """
    fig, ax = plt.subplots(figsize=(12, 1))
    
    lines = []
    labels = []

    # Add trajectory lines for legend
    lines.append(plt.Line2D([0], [0], color=colors[0], linewidth=3, alpha=0.7))
    labels.append('Marker LR (X)')
    lines.append(plt.Line2D([0], [0], color=colors[0], linewidth=3, linestyle='--'))
    labels.append('Predicted LR (X)')
    lines.append(plt.Line2D([0], [0], color=colors[1], linewidth=3, alpha=0.7))
    labels.append('Marker SI (Y)')
    lines.append(plt.Line2D([0], [0], color=colors[1], linewidth=3, linestyle='--'))
    labels.append('Predicted SI (Y)')
    lines.append(plt.Line2D([0], [0], color=colors[2], linewidth=3, alpha=0.7))
    labels.append('Marker AP (Z)')
    lines.append(plt.Line2D([0], [0], color=colors[2], linewidth=3, linestyle='--'))
    labels.append('Predicted AP (Z)')
    lines.append(plt.Line2D([0], [0], color=colors[3], linewidth=3))
    labels.append('Euclidean Error')

    # Create legend
    legend = ax.legend(lines, labels,
                       loc='center',
                       ncol=4,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       fontsize=24)

    # Hide axes
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'trajectory_legend.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join(save_path, 'trajectory_legend.pdf'),
                    bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def load_and_align_trajectories(markers_folder, tumor_folder):
    """
    Load marker and tumor prediction trajectories and ensure alignment.
    
    Parameters:
    markers_folder: Path to marker trajectory folder
    tumor_folder: Path to tumor prediction trajectory folder
    
    Returns:
    Dictionary containing aligned data, keyed by case ID
    """
    marker_files = [f for f in os.listdir(markers_folder) if f.endswith('.csv')]
    cases_data = {}
    
    for file in marker_files:
        match = re.match(r'(PT\d+_F\d+)\.csv', file)
        if not match:
            continue
            
        case_id = match.group(1)
        tumor_file = os.path.join(tumor_folder, f"{case_id}.csv")
        
        if not os.path.exists(tumor_file):
            print(f"Warning: Corresponding tumor prediction file {tumor_file} not found")
            continue

        # Load data
        marker_df = pd.read_csv(os.path.join(markers_folder, file))
        tumor_df = pd.read_csv(tumor_file)
        
        # Ensure data alignment by frame number
        min_frames = min(len(marker_df), len(tumor_df))
        marker_df = marker_df.head(min_frames)
        tumor_df = tumor_df.head(min_frames)
        
        cases_data[case_id] = {
            'marker': marker_df,
            'tumor': tumor_df
        }
    
    return cases_data


def calculate_errors(cases_data):
    """
    Calculate error metrics for each case.
    
    Parameters:
    cases_data: Dictionary containing aligned data
    
    Returns:
    DataFrame containing error metrics
    """
    error_data = []
    
    for case_id, data in cases_data.items():
        marker_df = data['marker']
        tumor_df = data['tumor']
        
        # Calculate Euclidean distance error
        euclidean_errors = np.sqrt(
            (marker_df['Cx'] - tumor_df['Cx']) ** 2 +
            (marker_df['Cy'] - tumor_df['Cy']) ** 2 +
            (marker_df['Cz'] - tumor_df['Cz']) ** 2
        )
        
        # Calculate axis-specific errors
        x_errors = np.abs(marker_df['Cx'] - tumor_df['Cx'])
        y_errors = np.abs(marker_df['Cy'] - tumor_df['Cy'])
        z_errors = np.abs(marker_df['Cz'] - tumor_df['Cz'])
        
        # Calculate statistical metrics
        error_stats = {
            'Case': case_id,
            'Mean_Euclidean_Error': np.mean(euclidean_errors),
            'Std_Euclidean_Error': np.std(euclidean_errors),
            'Max_Euclidean_Error': np.max(euclidean_errors),
            'Median_Euclidean_Error': np.median(euclidean_errors),
            'RMSE': np.sqrt(np.mean(euclidean_errors ** 2)),
            'Mean_LR_Error': np.mean(x_errors),
            'Std_LR_Error': np.std(x_errors),
            'Mean_SI_Error': np.mean(y_errors),
            'Std_SI_Error': np.std(y_errors),
            'Mean_AP_Error': np.mean(z_errors),
            'Std_AP_Error': np.std(z_errors),
            'Frames': len(marker_df)
        }
        error_data.append(error_stats)
    
    return pd.DataFrame(error_data)


def plot_trajectory_comparison(case_id, marker_df, tumor_df, save_path=None, show_legend=False):
    """
    Plot combined trajectory and error time series for a single case.
    
    Parameters:
    case_id: Case identifier
    marker_df: Marker trajectory DataFrame
    tumor_df: Tumor prediction trajectory DataFrame
    save_path: Directory to save plots
    show_legend: Whether to display legend on plot
    """
    fig, ax1 = plt.subplots(figsize=(20, 5))
    time = range(len(marker_df))

    # Position time series on primary y-axis
    ax1.plot(time, marker_df['Cx'], label='Marker LR (X)' if show_legend else "",
             color=colors[0], alpha=0.7, linewidth=2.5)
    ax1.plot(time, tumor_df['Cx'], label='Predicted LR (X)' if show_legend else "",
             color=colors[0], linestyle='--', linewidth=2.5)
    ax1.plot(time, marker_df['Cy'], label='Marker SI (Y)' if show_legend else "",
             color=colors[1], alpha=0.7, linewidth=2.5)
    ax1.plot(time, tumor_df['Cy'], label='Predicted SI (Y)' if show_legend else "",
             color=colors[1], linestyle='--', linewidth=2.5)
    ax1.plot(time, marker_df['Cz'], label='Marker AP (Z)' if show_legend else "",
             color=colors[2], alpha=0.7, linewidth=2.5)
    ax1.plot(time, tumor_df['Cz'], label='Predicted AP (Z)' if show_legend else "",
             color=colors[2], linestyle='--', linewidth=2.5)
    
    ax1.set_xlabel('Frame', fontsize=25)
    ax1.set_ylabel('Position (mm)', fontsize=25)

    # Error time series on secondary y-axis
    errors = np.sqrt(
        (marker_df['Cx'] - tumor_df['Cx']) ** 2 +
        (marker_df['Cy'] - tumor_df['Cy']) ** 2 +
        (marker_df['Cz'] - tumor_df['Cz']) ** 2
    )
    ax2 = ax1.twinx()
    ax2.plot(time, errors, label='Euclidean Error' if show_legend else "",
             color=colors[3], linewidth=3)
    ax2.set_ylabel('Euclidean Error (mm)', fontsize=25)

    # Adjust tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)

    # Add legend if requested
    if show_legend:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0.95),
                   ncol=4, fancybox=True, shadow=True,
                   framealpha=0.9, fontsize=20)

    plt.tight_layout()
    
    if save_path:
        filename = f'{case_id}_trajectory_comparison.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, f'{case_id}_trajectory_comparison.pdf'),
                    bbox_inches='tight')
    plt.show()
    plt.close()


def plot_summary_statistics(error_df, save_path=None):
    """
    Plot overall error statistics summary.
    
    Parameters:
    error_df: DataFrame containing error statistics
    save_path: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect all error data for visualization
    all_errors = []
    case_labels = []
    
    for _, row in error_df.iterrows():
        case_data = cases_data[row['Case']]
        errors = np.sqrt(
            (case_data['marker']['Cx'] - case_data['tumor']['Cx']) ** 2 +
            (case_data['marker']['Cy'] - case_data['tumor']['Cy']) ** 2 +
            (case_data['marker']['Cz'] - case_data['tumor']['Cz']) ** 2
        )
        all_errors.append(errors)
        case_labels.append(row['Case'])

    all_errors_flat = np.concatenate(all_errors)

    # Subplot 1: Mean error by case
    axes[0, 0].bar(range(len(error_df)), error_df['Mean_Euclidean_Error'],
                   color=colors[0], alpha=0.7)
    axes[0, 0].set_xlabel('Case', fontsize=25)
    axes[0, 0].set_ylabel('Mean Euclidean Error (mm)', fontsize=25)
    axes[0, 0].set_title('Mean Error by Case', fontsize=28, pad=15)
    axes[0, 0].set_xticks(range(len(error_df)))
    axes[0, 0].set_xticklabels(error_df['Case'], rotation=45, fontsize=22)
    axes[0, 0].tick_params(axis='y', labelsize=22)

    # Subplot 2: Mean error by axis
    lr_errors = error_df['Mean_LR_Error'].mean()
    si_errors = error_df['Mean_SI_Error'].mean()
    ap_errors = error_df['Mean_AP_Error'].mean()
    
    axes[0, 1].bar(['LR (X)', 'SI (Y)', 'AP (Z)'],
                   [lr_errors, si_errors, ap_errors],
                   color=[colors[0], colors[1], colors[2]], alpha=0.7)
    axes[0, 1].set_ylabel('Mean Error (mm)', fontsize=25)
    axes[0, 1].set_title('Mean Error by Axis', fontsize=28, pad=15)
    axes[0, 1].tick_params(axis='both', labelsize=22)

    # Subplot 3: Error distribution boxplot by case
    box_plot = axes[1, 0].boxplot(all_errors, labels=case_labels, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors[:len(all_errors)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_xlabel('Case', fontsize=25)
    axes[1, 0].set_ylabel('Euclidean Error (mm)', fontsize=25)
    axes[1, 0].set_title('Error Distribution by Case', fontsize=28, pad=15)
    axes[1, 0].tick_params(axis='both', labelsize=22)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Subplot 4: Overall error distribution histogram
    axes[1, 1].hist(all_errors_flat, bins=30, color=colors[4], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(all_errors_flat), color=colors[0], linestyle='--', linewidth=3,
                       label=f'Mean: {np.mean(all_errors_flat):.2f}mm')
    axes[1, 1].axvline(np.median(all_errors_flat), color=colors[1], linestyle='--', linewidth=3,
                       label=f'Median: {np.median(all_errors_flat):.2f}mm')
    axes[1, 1].set_xlabel('Euclidean Error (mm)', fontsize=25)
    axes[1, 1].set_ylabel('Frequency', fontsize=25)
    axes[1, 1].set_title('Overall Error Distribution', fontsize=28, pad=15)
    axes[1, 1].legend(fontsize=20, loc='upper right')
    axes[1, 1].tick_params(axis='both', labelsize=22)

    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'summary_statistics.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'summary_statistics.pdf'),
                    bbox_inches='tight')
    plt.show()
    plt.close()


def generate_summary_table(error_df, save_path=None):
    """
    Generate formatted summary tables of error statistics.
    
    Parameters:
    error_df: DataFrame containing error statistics
    save_path: Directory to save tables
    
    Returns:
    Tuple of (overall statistics DataFrame, formatted error DataFrame)
    """
    # Calculate overall statistics
    overall_stats = {
        'Metric': [
            'Overall Mean Euclidean Error ± Std',
            'Overall Mean LR Error ± Std',
            'Overall Mean SI Error ± Std',
            'Overall Mean AP Error ± Std',
            'Overall Max Euclidean Error',
            'Overall Median Euclidean Error',
            'Overall RMSE'
        ],
        'Value (mm)': [
            f"{error_df['Mean_Euclidean_Error'].mean():.2f} ± {error_df['Std_Euclidean_Error'].mean():.2f}",
            f"{error_df['Mean_LR_Error'].mean():.2f} ± {error_df['Std_LR_Error'].mean():.2f}",
            f"{error_df['Mean_SI_Error'].mean():.2f} ± {error_df['Std_SI_Error'].mean():.2f}",
            f"{error_df['Mean_AP_Error'].mean():.2f} ± {error_df['Std_AP_Error'].mean():.2f}",
            f"{error_df['Max_Euclidean_Error'].max():.2f}",
            f"{error_df['Median_Euclidean_Error'].median():.2f}",
            f"{error_df['RMSE'].mean():.2f}"
        ]
    }
    overall_df = pd.DataFrame(overall_stats)

    # Format error DataFrame with combined mean ± std columns
    formatted_error_df = error_df.copy()
    formatted_error_df['Mean_Euclidean_Error ± Std'] = formatted_error_df.apply(
        lambda row: f"{row['Mean_Euclidean_Error']:.2f} ± {row['Std_Euclidean_Error']:.2f}", axis=1
    )
    formatted_error_df['Mean_LR_Error ± Std'] = formatted_error_df.apply(
        lambda row: f"{row['Mean_LR_Error']:.2f} ± {row['Std_LR_Error']:.2f}", axis=1
    )
    formatted_error_df['Mean_SI_Error ± Std'] = formatted_error_df.apply(
        lambda row: f"{row['Mean_SI_Error']:.2f} ± {row['Std_SI_Error']:.2f}", axis=1
    )
    formatted_error_df['Mean_AP_Error ± Std'] = formatted_error_df.apply(
        lambda row: f"{row['Mean_AP_Error']:.2f} ± {row['Std_AP_Error']:.2f}", axis=1
    )

    # Drop individual mean and std columns
    columns_to_drop = ['Mean_Euclidean_Error', 'Std_Euclidean_Error', 
                      'Mean_LR_Error', 'Std_LR_Error',
                      'Mean_SI_Error', 'Std_SI_Error', 
                      'Mean_AP_Error', 'Std_AP_Error']
    formatted_error_df = formatted_error_df.drop(columns=columns_to_drop)

    # Format remaining numerical columns
    for col in ['Max_Euclidean_Error', 'Median_Euclidean_Error', 'RMSE']:
        formatted_error_df[col] = formatted_error_df[col].apply(lambda x: f"{x:.2f}")

    # Save LaTeX tables if requested
    if save_path:
        with open(os.path.join(save_path, 'error_summary.tex'), 'w') as f:
            f.write(formatted_error_df.to_latex(index=False))
        with open(os.path.join(save_path, 'overall_summary.tex'), 'w') as f:
            f.write(overall_df.to_latex(index=False))

    return overall_df, formatted_error_df


def perform_statistical_tests(cases_data, error_df, save_path=None):
    """
    Perform statistical tests on error data.
    
    Parameters:
    cases_data: Dictionary containing aligned data
    error_df: DataFrame containing error statistics
    save_path: Directory to save results
    
    Returns:
    DataFrame with statistical test results
    """
    # Collect all error data
    all_errors = []
    for case_id in cases_data.keys():
        case_data = cases_data[case_id]
        errors = np.sqrt(
            (case_data['marker']['Cx'] - case_data['tumor']['Cx']) ** 2 +
            (case_data['marker']['Cy'] - case_data['tumor']['Cy']) ** 2 +
            (case_data['marker']['Cz'] - case_data['tumor']['Cz']) ** 2
        )
        all_errors.extend(errors)
    
    all_errors = np.array(all_errors)

    # Normality test
    _, normality_p = stats.normaltest(all_errors)
    
    # Calculate confidence interval
    mean_error = np.mean(all_errors)
    sem = stats.sem(all_errors)
    ci = stats.t.interval(0.95, len(all_errors) - 1, loc=mean_error, scale=sem)

    # Create results table
    stats_results = pd.DataFrame({
        'Test': ['Shapiro-Wilk Normality Test', 
                 '95% Confidence Interval Lower', 
                 '95% Confidence Interval Upper'],
        'Value': [f"p = {normality_p:.4f}", 
                  f"{ci[0]:.2f} mm", 
                  f"{ci[1]:.2f} mm"]
    })

    if save_path:
        with open(os.path.join(save_path, 'statistical_tests.tex'), 'w') as f:
            f.write(stats_results.to_latex(index=False))

    return stats_results


def main():
    """Main analysis function."""
    # Set paths
    markers_folder = "path/to/marker/trajectories"
    tumor_folder = "path/to/tumor/prediction/trajectories"
    output_folder = "path/to/analysis/results"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Create standalone legend first
    print("Creating standalone legend...")
    create_standalone_legend(output_folder)

    # Load and align data
    global cases_data
    cases_data = load_and_align_trajectories(markers_folder, tumor_folder)
    
    if not cases_data:
        print("Error: No matching data files found")
        return
    
    print(f"Successfully loaded data for {len(cases_data)} cases")

    # Calculate error metrics
    error_df = calculate_errors(cases_data)

    # Plot trajectory comparisons without legends
    for case_id, data in cases_data.items():
        print(f"Generating trajectory comparison plot for {case_id}...")
        plot_trajectory_comparison(case_id, data['marker'], data['tumor'], 
                                 output_folder, show_legend=False)

    # Generate one example plot with legend for reference
    print("Generating one example plot with legend for reference...")
    first_case_id = list(cases_data.keys())[0]
    plot_trajectory_comparison(f"{first_case_id}_with_legend",
                             cases_data[first_case_id]['marker'],
                             cases_data[first_case_id]['tumor'],
                             output_folder, show_legend=True)

    # Generate summary statistics plots
    print("Generating overall statistics plot...")
    plot_summary_statistics(error_df, output_folder)

    # Generate summary tables
    print("Generating summary table...")
    overall_df, formatted_error_df = generate_summary_table(error_df, output_folder)

    # Perform statistical tests
    print("Performing statistical tests...")
    stats_results = perform_statistical_tests(cases_data, error_df, output_folder)

    # Print results
    print("\n=== Error Statistics Summary ===")
    print(formatted_error_df)
    print("\n=== Overall Statistics ===")
    print(overall_df)
    print("\n=== Statistical Test Results ===")
    print(stats_results)

    # Save results to CSV
    error_df.to_csv(os.path.join(output_folder, 'error_metrics.csv'), index=False)
    print(f"\nAnalysis complete! Results saved to {output_folder}")
    print("Note: Main plots are generated without legends. Use the standalone legend file for paper formatting.")


if __name__ == "__main__":
    main()

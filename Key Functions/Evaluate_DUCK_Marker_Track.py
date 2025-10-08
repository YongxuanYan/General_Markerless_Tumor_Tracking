import cv2
import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

# Assuming ModelArchitecture.py is in the same directory or accessible in the python path
from ModelArchitecture import DUCK_Net

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# --- Core Paths ---
# Anonymized: Please replace with your actual paths.
BASE_DIR = Path("/path/to/your/DUCK-Net/Input/Images")
OUTPUT_DIR = Path("./evaluation_output")
MODEL_PATH = Path("/path/to/saved/DuckNet")

# --- Experiment Naming ---
# This name is used to create unique output directories for this specific run.
EXPERIMENT_NAME = "DUCK-Net_Outputs"

# --- Derived Output Directories ---
# Directories are created automatically based on the experiment name.
MARKER_VISUALIZATION_DIR = OUTPUT_DIR / "Marker_Visualize"
COMPARISON_DIR = OUTPUT_DIR / f"comparison_{EXPERIMENT_NAME}"
SEGMENTATION_DIR = OUTPUT_DIR / f"segmentation_{EXPERIMENT_NAME}"
TRAJECTORY_DIR = OUTPUT_DIR / f"trajectory_{EXPERIMENT_NAME}"

# --- Model & Processing Parameters ---
IMG_SIZE = 256
PREDICTION_THRESHOLD = 0.5  # Confidence threshold to create a binary mask

# --- Evaluation Criteria (Root Mean Square Error in pixels) ---
RMSE_LEVELS = {
    'Excellent': (0, 3),
    'High': (3, 8),
    'Moderate': (8, 13),
    'Low': (13, float('inf'))
}

# --- Plotting Style ---
# Configuration for generating publication-quality figures.
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


# =============================================================================
# --- UTILITY FUNCTIONS ---
# =============================================================================

def dice_metric_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Computes the Dice coefficient loss.

    Args:
        y_true: The ground truth tensor.
        y_pred: The predicted tensor.
        smooth: A small constant to avoid division by zero.

    Returns:
        The calculated Dice loss as a TensorFlow tensor.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_coefficient


def load_segmentation_model(model_path: Path) -> tf.keras.Model:
    """
    Loads and compiles the DUCK-Net segmentation model.

    Args:
        model_path: The file path to the saved Keras model.

    Returns:
        A compiled TensorFlow/Keras model.
    """
    print(f"[INFO] Loading segmentation model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    custom_objects = {'dice_metric_loss': dice_metric_loss}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# =============================================================================
# --- MODEL INFERENCE ---
# =============================================================================

def get_tumor_centroid(
    model: tf.keras.Model,
    image_path: Path,
    save_mask: bool = False,
    mask_output_dir: Optional[Path] = None
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    Processes an image to find the tumor's centroid using the segmentation model.

    Args:
        model: The compiled DUCK-Net model.
        image_path: Path to the input image.
        save_mask: If True, saves the resulting segmentation mask.
        mask_output_dir: Directory to save the mask. Required if save_mask is True.

    Returns:
        A tuple containing:
        - The (x, y) coordinates of the tumor centroid, or None if not found.
        - The total processing time in milliseconds.
    """
    start_time = time.time()
    
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, 0.0

    original_shape = image.shape
    resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(normalized_image, -1), 0)

    # Model prediction
    predicted_mask_raw = model.predict(input_tensor, verbose=0)[0, ..., 0]
    binary_mask = (predicted_mask_raw > PREDICTION_THRESHOLD).astype(np.uint8)

    # Save mask if requested
    if save_mask and mask_output_dir:
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        mask_original_size = cv2.resize(
            binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        output_path = mask_output_dir / image_path.name
        cv2.imwrite(str(output_path), mask_original_size * 255)

    # Calculate centroid from moments
    centroid = None
    if np.sum(binary_mask) > 0:
        moments = cv2.moments(binary_mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            # Scale centroid coordinates back to original image size
            cx_orig = int(cx * original_shape[1] / IMG_SIZE)
            cy_orig = int(cy * original_shape[0] / IMG_SIZE)
            centroid = (cx_orig, cy_orig)

    end_time = time.time()
    inference_time_ms = 1000 * (end_time - start_time)

    return centroid, inference_time_ms


# =============================================================================
# --- ANALYSIS & METRICS ---
# =============================================================================

def analyze_trajectory_error(displacements: np.ndarray) -> Tuple[float, str]:
    """
    Calculates the Root Mean Square Error (RMSE) and determines the performance level.

    Args:
        displacements: A NumPy array of displacement errors.

    Returns:
        A tuple containing:
        - The calculated RMSE value.
        - The corresponding performance level string (e.g., 'Excellent').
    """
    if displacements.size == 0:
        return 0.0, 'Unknown'
        
    rmse = np.sqrt(np.mean(np.square(displacements)))
    for level, (low, high) in RMSE_LEVELS.items():
        if low <= rmse < high:
            return rmse, level
    return rmse, 'Unknown'


def calculate_level_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the percentage distribution of each performance level.

    Args:
        results: A list of result dictionaries, each with an 'rmse_level' key.

    Returns:
        A dictionary mapping each performance level to its percentage.
    """
    level_counts = {level: 0 for level in RMSE_LEVELS}
    for result in results:
        if result['rmse_level'] in level_counts:
            level_counts[result['rmse_level']] += 1
    
    total_results = len(results)
    if total_results == 0:
        return {level: 0.0 for level in RMSE_LEVELS}
        
    return {level: (count / total_results) * 100 for level, count in level_counts.items()}


# =============================================================================
# --- VISUALIZATION ---
# =============================================================================

def _plot_trajectory_axis(
    ax: plt.Axes,
    aligned_marker_coords: np.ndarray,
    tumor_coords: np.ndarray,
    axis_label: str,
    analysis_data: Dict[str, Any]
) -> None:
    """
    Helper function to plot marker and tumor trajectories for a single axis (X or Y).

    Args:
        ax: The Matplotlib axes object to plot on.
        aligned_marker_coords: NumPy array of aligned marker coordinates for the axis.
        tumor_coords: NumPy array of tumor coordinates for the axis.
        axis_label: The label for the coordinate axis ("X" or "Y").
        analysis_data: Dictionary containing RMSE and level information.
    """
    frame_numbers = np.arange(len(aligned_marker_coords))
    ax.plot(frame_numbers, aligned_marker_coords, 'b-', label='Marker', linewidth=1.5)
    ax.plot(frame_numbers, tumor_coords, 'r--', label='Tumor', linewidth=1.5)

    # A fixed Y-axis range can help in comparing plots across different sequences.
    ax.set_ylim(0, 256)
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel(f"{axis_label} Position (pixels)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    stats_text = (
        f"RMSE: {analysis_data[f'rmse_{axis_label.lower()}']:.2f} px\n"
        f"Level: {analysis_data[f'rmse_level_{axis_label.lower()}']}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))


def create_comparison_visualization(
    first_frame_path: Path,
    marker_trajectory: np.ndarray,
    tumor_trajectory: np.ndarray,
    analysis_data: Dict[str, Any],
    sequence_id: str
) -> Path:
    """
    Generates a publication-quality plot comparing marker and tumor trajectories.

    Args:
        first_frame_path: Path to the first image of the sequence for visualization.
        marker_trajectory: NumPy array of marker (x, y) coordinates.
        tumor_trajectory: NumPy array of tumor (x, y) coordinates.
        analysis_data: Dictionary of analysis results for this sequence.
        sequence_id: A unique identifier for the sequence.

    Returns:
        The path to the saved comparison plot image.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

    # Left Panel: Example Marker Image
    ax0 = plt.subplot(gs[:, 0])
    if first_frame_path.exists():
        ax0.imshow(plt.imread(first_frame_path), cmap='gray')
    ax0.set_title("First Frame of Sequence")
    ax0.axis('off')

    # Top-Right Panel: X-coordinate Trajectory
    ax1 = plt.subplot(gs[0, 1])
    _plot_trajectory_axis(
        ax1,
        np.array(analysis_data['aligned_marker_x']),
        tumor_trajectory[:, 0],
        'X',
        analysis_data
    )
    
    # Bottom-Right Panel: Y-coordinate Trajectory
    ax2 = plt.subplot(gs[1, 1])
    _plot_trajectory_axis(
        ax2,
        np.array(analysis_data['aligned_marker_y']),
        tumor_trajectory[:, 1],
        'Y',
        analysis_data
    )
    
    # Add a shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))

    # Main Title
    fig.suptitle(
        f"Sequence: {sequence_id}\n"
        f"Overall RMSE: {analysis_data['rmse']:.2f} px | Level: {analysis_data['rmse_level']}",
        fontsize=16
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = COMPARISON_DIR / f"{sequence_id}_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def create_summary_performance_chart(results_df: pd.DataFrame) -> None:
    """
    Creates and saves a bar chart summarizing the RMSE for all sequences.

    Args:
        results_df: DataFrame containing the analysis results for all sequences.
    """
    if results_df.empty:
        print("[WARNING] Results DataFrame is empty. Skipping summary chart creation.")
        return

    plt.figure(figsize=(12, 8))
    
    results_df = results_df.sort_values('rmse', ascending=True)
    y_pos = np.arange(len(results_df))

    level_color_map = {
        'Excellent': 'green',
        'High': 'blue',
        'Moderate': 'orange',
        'Low': 'red'
    }
    colors = results_df['rmse_level'].map(level_color_map).fillna('gray')

    plt.barh(y_pos, results_df['rmse'], color=colors, alpha=0.7)

    patches = [mpatches.Patch(color=color, label=f'{level} ({v[0]} ≤ RMSE < {v[1]})')
               for level, (color, v) in zip(level_color_map.keys(), RMSE_LEVELS.values())]

    plt.legend(handles=patches, loc='lower right')
    plt.yticks(y_pos, results_df['sequence_id'])
    plt.xlabel('Overall RMSE (pixels)')
    plt.title('Tumor Tracking Performance Summary (Max of X/Y RMSE)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    summary_path = COMPARISON_DIR / "rmse_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[INFO] Summary performance chart saved to: {summary_path}")


# =============================================================================
# --- MAIN WORKFLOW ---
# =============================================================================

def run_evaluation_pipeline():
    """
    Main function to execute the full evaluation workflow.
    It processes patient data, segments tumors, analyzes tracking error,
    and generates visualizations.
    """
    segmentation_model = load_segmentation_model(MODEL_PATH)
    all_results = []
    
    # Create parent output directories
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    patient_dirs = natsorted(list(BASE_DIR.glob("PT*")))
    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        field_dirs = natsorted(list(patient_dir.glob("F*")))
        for field_dir in field_dirs:
            for device in ["grab0", "grab1"]:
                device_dir = field_dir / device
                if not device_dir.exists():
                    continue

                sequence_id = f"{patient_dir.name}_{field_dir.name}_{device}"
                marker_csv_path = OUTPUT_DIR / f"{sequence_id}.csv"

                if not marker_csv_path.exists():
                    print(f"[WARNING] Marker trajectory file not found, skipping: {marker_csv_path}")
                    continue

                marker_df = pd.read_csv(marker_csv_path)
                marker_traj = list(zip(marker_df['marker_x'], marker_df['marker_y']))
                image_files = natsorted(list(device_dir.glob("*.*")))

                if len(image_files) != len(marker_traj):
                    print(f"[WARNING] Mismatch in frame count for {sequence_id}. "
                          f"Images: {len(image_files)}, Trajectory points: {len(marker_traj)}. Skipping.")
                    continue

                tumor_traj = []
                total_inference_time = 0
                for img_path in tqdm(image_files, desc=f"Segmenting {sequence_id}", leave=False):
                    mask_output_dir = SEGMENTATION_DIR / patient_dir.name / field_dir.name / device
                    centroid, time_ms = get_tumor_centroid(
                        segmentation_model, img_path, save_mask=True, mask_output_dir=mask_output_dir
                    )
                    
                    total_inference_time += time_ms
                    
                    # Impute missing centroids using the last known position
                    if centroid:
                        tumor_traj.append(centroid)
                    else:
                        fallback_pos = tumor_traj[-1] if tumor_traj else marker_traj[0]
                        tumor_traj.append(fallback_pos)
                
                avg_time = total_inference_time / len(image_files) if image_files else 0
                print(f"Average processing time for {sequence_id}: {avg_time:.2f} ms/frame")

                marker_arr = np.array(marker_traj)
                tumor_arr = np.array(tumor_traj)

                # Align trajectories by centering them around their mean
                aligned_marker_x = marker_arr[:, 0] - (np.mean(marker_arr[:, 0]) - np.mean(tumor_arr[:, 0]))
                aligned_marker_y = marker_arr[:, 1] - (np.mean(marker_arr[:, 1]) - np.mean(tumor_arr[:, 1]))
                
                displacements_x = np.abs(aligned_marker_x - tumor_arr[:, 0])
                displacements_y = np.abs(aligned_marker_y - tumor_arr[:, 1])

                rmse_x, level_x = analyze_trajectory_error(displacements_x)
                rmse_y, level_y = analyze_trajectory_error(displacements_y)
                
                # Overall error is the maximum of the errors in each axis
                overall_rmse = max(rmse_x, rmse_y)
                _, overall_level = analyze_trajectory_error(np.array([overall_rmse]))

                analysis_data = {
                    'sequence_id': sequence_id,
                    'rmse_x': rmse_x,
                    'rmse_level_x': level_x,
                    'rmse_y': rmse_y,
                    'rmse_level_y': level_y,
                    'rmse': overall_rmse,
                    'rmse_level': overall_level,
                    'aligned_marker_x': aligned_marker_x.tolist(),
                    'aligned_marker_y': aligned_marker_y.tolist()
                }

                comparison_path = create_comparison_visualization(
                    image_files[0], marker_arr, tumor_arr, analysis_data, sequence_id
                )
                
                all_results.append({**analysis_data, 'comparison_path': str(comparison_path)})

    if not all_results:
        print("[WARNING] Processing completed, but no results were generated.")
        return
        
    # Save results to a CSV file
    results_df = pd.DataFrame(all_results)
    full_trajectory_path = TRAJECTORY_DIR / "all_sequence_results.csv"
    results_df.to_csv(full_trajectory_path, index=False)
    print(f"\n[SUCCESS] Full analysis data saved to: {full_trajectory_path}")

    # Print summary statistics
    level_stats = calculate_level_statistics(all_results)
    print("\n--- RMSE Level Distribution ---")
    for level, percent in level_stats.items():
        print(f"{level:>10}: {percent:.1f}%")
    print("-----------------------------\n")

    # Create final summary chart
    create_summary_performance_chart(results_df)


if __name__ == "__main__":
    print("[INFO] Starting tumor tracking evaluation pipeline...")
    run_evaluation_pipeline()
    print("[INFO] Pipeline execution finished.")

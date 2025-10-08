import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# Anonymized: Please replace with your actual paths.
BASE_DATA_DIR = Path("/path/to/your/PatientData/CFI")
OUTPUT_DIR = Path("/path/to/your/Markers/Trajectories")

# Derived output paths
MARKER_VISUALIZATION_DIR = OUTPUT_DIR / "Markers"
PROCESSED_LOG_FILE = OUTPUT_DIR / "processed_sequences.txt"

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(8, 8),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
)

# =============================================================================
# --- UTILITY FUNCTIONS ---
# =============================================================================

def load_processed_sequences() -> set:
    """Loads the set of sequence IDs that have already been processed."""
    if not PROCESSED_LOG_FILE.exists():
        return set()
    with open(PROCESSED_LOG_FILE, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def mark_sequence_as_processed(sequence_id: str) -> None:
    """Appends a sequence ID to the log of processed sequences."""
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(f"{sequence_id}\n")

def get_user_selected_point(image: np.ndarray) -> tuple:
    """
    Displays an image and prompts the user to click on a point.

    Args:
        image: The grayscale image to display.

    Returns:
        A tuple (x, y) of the selected coordinates.
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title("Select the fiducial marker (black dot), then close this window to continue.")
    ax.axis('on')

    print("\nPlease select the marker in the displayed window...")
    # ginput(1, timeout=0) waits indefinitely for one click
    selected_point = plt.ginput(1, timeout=0)[0]
    plt.close(fig)
    print(f"Point selected at: ({selected_point[0]:.1f}, {selected_point[1]:.1f})")
    return selected_point

def save_marked_image(image_path: Path, point: tuple, output_path: Path) -> bool:
    """
    Saves a copy of an image with a green square marking a specified point.

    Args:
        image_path: Path to the original image.
        point: The (x, y) coordinate to mark.
        output_path: Path to save the marked image.

    Returns:
        True if successful, False otherwise.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    x, y = int(point[0]), int(point[1])
    # Draw a 10x10 green rectangle centered on the point
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), image)
    return True

# =============================================================================
# --- CORE TRACKING LOGIC ---
# =============================================================================

def track_markers_in_sequence(image_files: list, sequence_id: str) -> pd.DataFrame | None:
    """
    Performs optical flow tracking on a single sequence of images.

    Args:
        image_files: A sorted list of image file paths.
        sequence_id: A unique identifier for the sequence.

    Returns:
        A DataFrame with the trajectory data, or None if tracking fails.
    """
    marker_output_dir = MARKER_VISUALIZATION_DIR / sequence_id
    marker_output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"Error: Could not read the first frame: {image_files[0]}")
        return None

    # Use the red channel for tracking, as fiducials are often dark
    first_frame_red_channel = first_frame[:, :, 2]  # OpenCV is BGR, so index 2 is Red

    try:
        initial_point = get_user_selected_point(first_frame_red_channel)
    except Exception as e:
        print(f"Error during point selection: {e}. Skipping sequence.")
        return None

    # Save visualization of the selected point on the first frame
    first_frame_output_path = marker_output_dir / image_files[0].name
    save_marked_image(image_files[0], initial_point, first_frame_output_path)

    # Prepare for optical flow. Invert the image so the dark marker becomes bright.
    prev_image_inverted = 255 - first_frame_red_channel
    prev_point = np.array([initial_point], dtype=np.float32).reshape(-1, 1, 2)
    trajectory = [initial_point]

    for image_path in tqdm(image_files[1:], desc=f"Tracking {sequence_id}", leave=False):
        current_frame = cv2.imread(str(image_path))
        if current_frame is None:
            print(f"Warning: Could not read frame {image_path.name}. Using last known position.")
            trajectory.append(trajectory[-1])
            continue
            
        current_image_inverted = 255 - current_frame[:, :, 2]
        
        # Calculate optical flow
        next_point, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_image_inverted, current_image_inverted, prev_point, None, **LK_PARAMS
        )

        # Update trajectory and reference point
        if status[0] == 1 and next_point is not None:
            current_pos = tuple(next_point.ravel())
            trajectory.append(current_pos)
            prev_point = next_point.reshape(-1, 1, 2)
        else: # If tracking is lost, use the last known position
            trajectory.append(trajectory[-1])
        
        # Update reference image for the next frame
        prev_image_inverted = current_image_inverted

    # Create a DataFrame from the tracked trajectory
    result_df = pd.DataFrame(
        [{'frame': idx, 'marker_x': x, 'marker_y': y} for idx, (x, y) in enumerate(trajectory)]
    )
    return result_df

# =============================================================================
# --- MAIN EXECUTION ---
# =============================================================================

def process_all_patient_data():
    """
    Main function to iterate through all patient data, track fiducial markers,
    and save the results.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    processed_sequences = load_processed_sequences()
    patient_dirs = natsorted(list(BASE_DATA_DIR.glob("PT*")))

    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        field_dirs = natsorted(list(patient_dir.glob("F*")))
        for field_dir in field_dirs:
            for device in ["grab0", "grab1"]:
                device_dir = field_dir / device
                if not device_dir.is_dir():
                    continue

                sequence_id = f"{patient_dir.name}/{field_dir.name}/{device}"
                if sequence_id in processed_sequences:
                    print(f"Skipping already processed sequence: {sequence_id}")
                    continue

                image_files = natsorted(list(device_dir.glob("*.[pjt][npi][gf]")))
                if not image_files:
                    print(f"No images found in sequence: {sequence_id}")
                    continue

                print(f"\nNow processing sequence: {sequence_id} ({len(image_files)} images)")
                
                sequence_df = track_markers_in_sequence(image_files, sequence_id)
                if sequence_df is None:
                    print(f">> Failed to process sequence {sequence_id}. Skipping.")
                    continue

                # Add metadata and save the results to a CSV file
                sequence_df['patient'] = patient_dir.name
                sequence_df['field'] = field_dir.name
                sequence_df['device'] = device
                
                output_filename = f"{sequence_id.replace('/', '_')}.csv"
                output_path = OUTPUT_DIR / output_filename
                sequence_df.to_csv(output_path, index=False)
                
                mark_sequence_as_processed(sequence_id)
                print(f">> Trajectory saved to: {output_path}")

    print("\nAll sequences have been processed!")


if __name__ == "__main__":
    process_all_patient_data()

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import glob
import shutil


"""
Optical flow-based tracking of fiducial markers in X-ray images.
Generates processed list for skipping, trajectory CSV files, and marked images with green bounding boxes.
"""

# Configuration parameters
BASE_DIR = "path/to/patient/data"  # Base directory containing patient data
OUTPUT_DIR = "path/to/trajectory/output"  # Output directory for trajectories
MARKERS_DIR = os.path.join(OUTPUT_DIR, "Markers")  # Directory for marked images
PROCESSED_LOG = os.path.join(OUTPUT_DIR, "processed_sequences.txt")  # Log file for processed sequences


def load_processed_sequences():
    """Load list of already processed sequences."""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_sequence_processed(sequence_id):
    """Mark a sequence as processed."""
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{sequence_id}\n")


def get_user_selected_point(r_channel):
    """Display R channel and allow user to manually select marker point."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(r_channel, cmap='gray')
    ax.set_title("Please select marker (black dot) and then close the window")
    ax.axis('on')

    print("\nPlease select marker dot")
    selected_point = plt.ginput(1, timeout=0)[0]
    plt.close()
    return selected_point  # Return (x,y) tuple


def save_marked_image(img_path, point, output_path):
    """Save image with marked point."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y = int(point[0]), int(point[1])

    # Draw 10x10 green bounding box
    cv2.rectangle(img_rgb, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)

    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return True


def track_single_sequence(img_files, sequence_id):
    """Process single sequence and save marked images."""
    marker_output_dir = os.path.join(MARKERS_DIR, sequence_id.replace('/', '\\'))
    os.makedirs(marker_output_dir, exist_ok=True)

    # Read first frame (extract R channel)
    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"Cannot read first frame: {img_files[0]}")
        return None

    r_channel = first_frame[:, :, 0]  # OpenCV uses BGR order, index 2 is R channel

    # User manually selects marker point
    try:
        initial_point = get_user_selected_point(r_channel)
        print(f"Selected point ({initial_point[0]:.1f}, {initial_point[1]:.1f})")
    except Exception as e:
        print(f"Selection failed: {e}")
        return None

    # Save first marked image
    first_output = os.path.join(marker_output_dir, os.path.basename(img_files[0]))
    save_marked_image(img_files[0], initial_point, first_output)

    # Prepare optical flow tracking (using inverted R channel)
    inverted_r = 255 - r_channel  # Invert to make black dots white
    trajectory = [initial_point]
    prev_pt = np.array([initial_point], dtype=np.float32).reshape(-1, 1, 2)
    prev_img = inverted_r

    lk_params = dict(
        winSize=(8, 8),
        maxLevel=2,  # Increase pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
    )

    # Process subsequent frames
    for i, path in enumerate(tqdm(img_files[1:], desc="Tracking", leave=False)):
        curr_frame = cv2.imread(path)
        if curr_frame is None:
            print(f"Warning: cannot read {os.path.basename(path)}")
            trajectory.append(trajectory[-1])  # Use previous position
            if trajectory:
                save_marked_image(path, trajectory[-1],
                                os.path.join(marker_output_dir, os.path.basename(path)))
            continue

        curr_img = 255 - curr_frame[:, :, 0]  # Inverted R channel
        curr_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img, curr_img, prev_pt, None, **lk_params)

        # Update trajectory
        if status[0] == 1:
            x, y = curr_pt.ravel()
            trajectory.append((x, y))
        else:
            trajectory.append(trajectory[-1])

        # Save current marked image
        save_marked_image(path, trajectory[-1],
                        os.path.join(marker_output_dir, os.path.basename(path)))

        # Update references
        prev_img = curr_img
        prev_pt = curr_pt if status[0] == 1 else prev_pt

    # Save sequence results
    result_df = pd.DataFrame([
        {'frame': idx, 'marker_x': x, 'marker_y': y}
        for idx, (x, y) in enumerate(trajectory)
    ])
    return result_df


def process_all_sequences():
    """Process all patient data sequences."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MARKERS_DIR, exist_ok=True)
    processed = load_processed_sequences()
    patient_dirs = natsorted(glob.glob(os.path.join(BASE_DIR, "PT*")))

    for p_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = os.path.basename(p_dir)

        # Process F folders
        for f_dir in natsorted(glob.glob(os.path.join(p_dir, "F*"))):
            field_id = os.path.basename(f_dir)

            # Process both grab folders
            for grab_dir in [os.path.join(f_dir, "grab0"), os.path.join(f_dir, "grab1")]:
                if not os.path.exists(grab_dir):
                    continue

                # Build unique sequence ID
                device_id = os.path.basename(grab_dir)
                sequence_id = f"{patient_id}/{field_id}/{device_id}"

                # Skip already processed sequences
                if sequence_id in processed:
                    print(f"Skipping processed sequence: {sequence_id}")
                    continue

                # Get image files
                img_files = natsorted(
                    glob.glob(os.path.join(grab_dir, "*.png")) +
                    glob.glob(os.path.join(grab_dir, "*.tif")) +
                    glob.glob(os.path.join(grab_dir, "*.jpg")),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )

                if not img_files:
                    print(f"No images found in sequence: {sequence_id}")
                    continue

                print(f"\nProcessing sequence: {sequence_id}")
                print(f"Number of images: {len(img_files)}")

                # Process current sequence
                seq_df = track_single_sequence(img_files, sequence_id)
                if seq_df is None:
                    print(">> Processing failed, skipping")
                    continue

                # Add metadata and save
                seq_df['patient'] = patient_id
                seq_df['field'] = field_id
                seq_df['device'] = device_id

                output_file = os.path.join(OUTPUT_DIR, f"{sequence_id.replace('/', '_')}.csv")
                seq_df.to_csv(output_file, index=False)
                mark_sequence_processed(sequence_id)
                print(f">> Saved to: {output_file}")

    print("\nAll sequences processed successfully!")


if __name__ == "__main__":
    process_all_sequences()

import pandas as pd
import ast
import re
from pathlib import Path
from typing import List, Tuple

# --- CONFIGURATION ---
# Anonymized: Please replace with your actual paths.
SOURCE_TRAJECTORIES_CSV = Path("/path/to/your/full_trajectories.csv")
MARKER_DATA_DIR = Path("/path/to/your/Markers/Trajectories")
OUTPUT_DIR = Path("/path/to/your/Splitted/TumorCenters_Trajectories")

# --- DEFAULT METADATA ---
# These values are used if a corresponding marker file or its columns are missing.
DEFAULT_METADATA = {
    'X tube x': 0.0,
    'X tube y': 0.0,
    'X tube z': 0.0,
    'IPEL': 233.38,
    'OID': 2444.0,
    'CouchAngle': 0.0
}

def parse_trajectory_string(traj_str: str) -> List[Tuple[int, int]]:
    """
    Parses a string representation of a trajectory into a list of coordinate tuples.
    Handles both literal list format and complex string formats.

    Args:
        traj_str: The string containing the trajectory data.

    Returns:
        A list of (x, y) integer tuples.
    """
    try:
        # First, try to evaluate the string as a Python literal (e.g., "[(1, 2), (3, 4)]")
        return ast.literal_eval(traj_str)
    except (ValueError, SyntaxError):
        # If that fails, use regex to find all coordinate pairs
        coords = re.findall(r'\((\d+),\s*(\d+)\)', traj_str)
        return [(int(x), int(y)) for x, y in coords]

def split_combined_trajectory_file():
    """
    Reads a CSV file containing combined trajectory data for multiple sequences,
    splits each sequence into its own CSV file, and merges it with corresponding
    marker metadata.
    """
    if not SOURCE_TRAJECTORIES_CSV.is_file():
        print(f"Error: Source trajectory file not found at '{SOURCE_TRAJECTORIES_CSV}'")
        return

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    source_df = pd.read_csv(SOURCE_TRAJECTORIES_CSV)

    for _, row in source_df.iterrows():
        patient = str(row['patient'])
        field = str(row['field'])
        device = str(row['device'])

        # Construct the output filename
        sequence_filename = f"{patient}_{field}_{device}.csv"
        output_path = OUTPUT_DIR / sequence_filename

        # Parse the 'tumor_traj' column into a list of coordinates
        tumor_trajectory = parse_trajectory_string(row['tumor_traj'])

        # Create a new DataFrame for the individual sequence
        trajectory_records = [{
            'frame': frame_idx,
            'tumor_x': x,
            'tumor_y': y
        } for frame_idx, (x, y) in enumerate(tumor_trajectory)]
        
        sequence_df = pd.DataFrame(trajectory_records)

        # Attempt to load corresponding marker metadata
        marker_file_path = MARKER_DATA_DIR / sequence_filename
        metadata_to_add = DEFAULT_METADATA.copy()

        if marker_file_path.exists():
            marker_df = pd.read_csv(marker_file_path)
            if not marker_df.empty and 'X tube x' in marker_df.columns:
                first_row = marker_df.iloc[0]
                metadata_to_add['X tube x'] = first_row['X tube x']
                metadata_to_add['X tube y'] = first_row['X tube y']
                metadata_to_add['X tube z'] = first_row['X tube z']
                metadata_to_add['IPEL'] = first_row.get('IPEL', DEFAULT_METADATA['IPEL'])
                metadata_to_add['OID'] = first_row.get('OID', DEFAULT_METADATA['OID'])
                metadata_to_add['CouchAngle'] = first_row.get('CouchAngle', DEFAULT_METADATA['CouchAngle'])
            else:
                print(f"Warning: Marker file '{sequence_filename}' is empty or missing required columns. Using default metadata.")
        else:
            print(f"Warning: Marker file not found for '{sequence_filename}'. Using default metadata.")
        
        # Add metadata columns to the sequence DataFrame
        for col, value in metadata_to_add.items():
            sequence_df[col] = value
            
        # Ensure consistent column order
        column_order = ['frame', 'tumor_x', 'tumor_y'] + list(DEFAULT_METADATA.keys())
        sequence_df = sequence_df[column_order]

        # Save the new CSV file
        sequence_df.to_csv(output_path, index=False)
        print(f"Successfully saved: {sequence_filename} ({len(tumor_trajectory)} frames)")

    print("\nProcessing complete. All files have been split.")


if __name__ == "__main__":
    split_combined_trajectory_file()

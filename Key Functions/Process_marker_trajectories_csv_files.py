import pandas as pd
from pathlib import Path

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# --- Directory Path ---
# Anonymized: Please replace with the path to your trajectory CSV files.
TRAJECTORY_DATA_DIR = Path("/path/to/your/PatientData/Trajectories")

# --- Machine-Specific Metadata ---
# Coordinates for the X-ray tube based on the machine ID.
MACHINE_COORDINATES = {
    '1': [-1332, 844, -1504],
    '2': [1332, 844, -1504],
    '3': [1332, -844, -1504],
    '4': [-1332, -844, -1504]
}

# --- Default Imaging Parameters ---
DEFAULT_IPEL = 233.38
DEFAULT_OID = 2444

# --- Data Cleanup ---
# Columns to remove from the CSV files if they exist.
COLUMNS_TO_DROP = ['patient', 'field', 'device']

# =============================================================================
# --- CORE FUNCTION ---
# =============================================================================

def add_machine_metadata_to_files(directory: Path) -> None:
    """
    Interactively processes all CSV files in a given directory to add machine-
    specific metadata.

    For each CSV file, it:
    1. Removes unnecessary identification columns.
    2. Prompts the user to specify the machine ID (1-4).
    3. Adds columns for X-ray tube coordinates, IPEL, and OID based on the input.
    4. Overwrites the original file with the updated data.

    Args:
        directory: The path to the folder containing the CSV trajectory files.
    """
    if not directory.is_dir():
        print(f"Error: Directory not found at '{directory}'")
        return

    csv_paths = sorted(list(directory.glob('*.csv')))
    if not csv_paths:
        print(f"No CSV files found in '{directory}'.")
        return

    print(f"Found {len(csv_paths)} CSV files to process in '{directory}'.\n")

    for file_path in csv_paths:
        try:
            df = pd.read_csv(file_path)

            # 1. Drop specified columns if they exist.
            df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')

            # 2. Interactively ask the user for the machine ID.
            while True:
                prompt = f"For file '{file_path.name}', which machine was used? (Enter 1, 2, 3, or 4): "
                machine_id = input(prompt).strip()
                if machine_id in MACHINE_COORDINATES:
                    break
                else:
                    print("Invalid input. Please enter a number between 1 and 4.")

            # 3. Add new columns based on the selected machine and defaults.
            x_tube_coords = MACHINE_COORDINATES[machine_id]
            df['X tube x'] = x_tube_coords[0]
            df['X tube y'] = x_tube_coords[1]
            df['X tube z'] = x_tube_coords[2]
            df['IPEL'] = DEFAULT_IPEL
            df['OID'] = DEFAULT_OID

            # 4. Save the modified DataFrame, overwriting the original file.
            df.to_csv(file_path, index=False)
            print(f"-> Successfully processed and saved: {file_path.name}\n")

        except Exception as e:
            print(f"An error occurred while processing {file_path.name}: {e}")

# =============================================================================
# --- SCRIPT EXECUTION ---
# =============================================================================

if __name__ == "__main__":
    add_machine_metadata_to_files(TRAJECTORY_DATA_DIR)
    print("All files have been processed.")

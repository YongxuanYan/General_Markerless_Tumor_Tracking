import os
import pandas as pd


def add_couch_angle_info(folder_path):
    """
    Add couch angle information to CSV files in the specified folder.
    
    Parameters:
    folder_path: Path to the folder containing CSV files
    """
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print(f"Found {len(csv_files)} CSV files in the folder.")

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        # Read CSV file
        df = pd.read_csv(file_path)

        # Check if CouchAngle column already exists
        if 'CouchAngle' in df.columns:
            print(f"File {csv_file} already contains CouchAngle information. Skipping.")
            continue

        # Prompt user for couch angle
        while True:
            try:
                user_input = input(f"Enter couch angle for {csv_file} (range [0-360)): ").strip()
                couch_angle = float(user_input)
                if 0 <= couch_angle < 360:
                    break
                else:
                    print("Invalid input. Please enter a value between 0 and 360 (exclusive of 360).")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        # Add CouchAngle column
        df['CouchAngle'] = couch_angle

        # Save the modified file (overwriting the original)
        df.to_csv(file_path, index=False)

        print(f"Couch angle added and file saved: {csv_file}")


def main():
    """Main function to execute the couch angle addition process."""
    folder_path = "path/to/trajectory/data"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return
    
    add_couch_angle_info(folder_path)
    print("Couch angle addition process completed.")


if __name__ == "__main__":
    main()

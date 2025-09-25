import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate_around_z(x, y, z, angle_degrees):
    """
    Rotate point coordinates around Z-axis.

    Parameters:
    x, y, z: Point coordinates
    angle_degrees: Rotation angle in degrees

    Returns:
    Rotated coordinates as numpy array
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    return np.array([
        x * cos_theta - y * sin_theta,
        x * sin_theta + y * cos_theta,
        z
    ])


def compute_image_point(S, P_img, OID, resolution, IPEL):
    """
    Calculate 3D coordinates of point on imaging plane.

    Parameters:
    S: 3D array, X-ray source coordinates
    P_img: 2D tuple, point coordinates on imaging plane (px, py) in pixels
    OID: Scalar, object to image distance (mm)
    resolution: Image resolution in pixels
    IPEL: Imaging plane edge length (mm)

    Returns:
    3D array, imaging point coordinates
    """
    O = np.array([0.0, 0.0, 0.0])
    D_vec = O - S  # Vector from S to O
    dist_SO = np.linalg.norm(D_vec)
    if dist_SO < 1e-10:
        raise ValueError("X-ray source located at origin, cannot define direction.")
    D_unit = D_vec / dist_SO

    # Calculate imaging plane center
    C_plane = O + OID * D_unit

    # Calculate X-axis vector
    dx, dy, dz = D_unit
    L_xy = np.sqrt(dx * dx + dy * dy)
    if L_xy < 1e-10:
        # If D_unit has zero X and Y components, use (1,0,0) as X-axis
        X_vec = np.array([1.0, 0.0, 0.0])
    else:
        X_vec = np.array([dy, -dx, 0.0]) / L_xy

    # Calculate Y-axis vector: Y_vec = - (D_unit × X_vec)
    Y_vec = np.cross(D_unit, X_vec)

    # Convert pixel coordinates to millimeter coordinates
    pixel_size = IPEL / resolution  # Pixel size in mm
    px_mm, py_mm = (P_img[0] - resolution / 2) * pixel_size, (P_img[1] - resolution / 2) * pixel_size

    # Calculate 3D coordinates of imaging point
    point_3d = C_plane + px_mm * X_vec + py_mm * Y_vec
    return point_3d


def compute_intersection(XSA, XSB, PA, PB, OID, CouchAngle, resolution, IPEL):
    """
    Calculate intersection point or midpoint of closest approach between two X-rays.

    Parameters:
    XSA: 3D array, first X-ray source coordinates (mm)
    XSB: 3D array, second X-ray source coordinates (mm)
    PA: 2D tuple, point coordinates on first imaging plane (px, py) in pixels
    PB: 2D tuple, point coordinates on second imaging plane (px, py) in pixels
    OID: Scalar, object to image distance (mm)
    CouchAngle: Couch rotation angle (degrees)
    resolution: Image resolution in pixels
    IPEL: Imaging plane edge length (mm)

    Returns:
    3D array, intersection point C coordinates
    """
    # Rotate X-ray source points
    XSA_rotated = rotate_around_z(XSA[0], XSA[1], XSA[2], -CouchAngle)
    XSB_rotated = rotate_around_z(XSB[0], XSB[1], XSB[2], -CouchAngle)

    # Calculate 3D points on imaging planes
    IPA = compute_image_point(XSA_rotated, PA, OID, resolution, IPEL)
    IPB = compute_image_point(XSB_rotated, PB, OID, resolution, IPEL)

    # Calculate ray direction vectors and normalize
    u_dir = IPA - XSA_rotated
    v_dir = IPB - XSB_rotated
    u_norm = np.linalg.norm(u_dir)
    v_norm = np.linalg.norm(v_dir)
    if u_norm < 1e-10 or v_norm < 1e-10:
        raise ValueError("Ray length is zero, cannot calculate.")
    u_unit = u_dir / u_norm
    v_unit = v_dir / v_norm

    A0 = XSA_rotated
    B0 = XSB_rotated
    W0 = A0 - B0

    d = np.dot(u_unit, v_unit)

    # Check if rays are parallel
    if abs(1 - d * d) < 1e-10:
        # Handle parallel rays
        if np.allclose(u_unit, v_unit):
            # Same direction
            proj = np.dot(B0 - A0, u_unit)
            if proj >= 0:
                P = A0 + proj * u_unit
                Q = B0
            else:
                P = A0
                Q = B0 + (-proj) * u_unit
        elif np.allclose(u_unit, -v_unit):
            # Opposite direction
            lambda0 = np.dot(B0 - A0, u_unit)
            if lambda0 >= 0:
                P = A0 + lambda0 * u_unit
                Q = B0
            else:
                P = A0
                Q = B0
        else:
            # Other cases (should not occur theoretically), use source points
            P = A0
            Q = B0
    else:
        # Handle non-parallel rays
        dot_u = np.dot(W0, u_unit)
        dot_v = np.dot(W0, v_unit)
        t = (dot_v * d - dot_u) / (1 - d * d)
        s = (dot_u * d - dot_v) / (1 - d * d)

        if s < 0:
            s = -s
        if t < 0:
            t = -t

        P = A0 + t * u_unit
        Q = B0 + s * v_unit

    C = (P + Q) / 2

    return C


def process_marker_files(folder_path, resolution):
    """
    Batch process marker files to calculate 3D trajectory points.

    Parameters:
    folder_path: Path to folder containing CSV files
    resolution: Image resolution in pixels

    Returns:
    None, but generates new CSV files in the same folder
    """
    # Get all CSV files in folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Group files by patient and treatment field
    file_groups = {}
    for file in csv_files:
        # Extract PT and F information using regex
        match = re.match(r'(PT\d+_F\d+)_grab(\d+)\.csv', file)
        if match:
            base_name = match.group(1)
            grab_num = match.group(2)

            if base_name not in file_groups:
                file_groups[base_name] = {}

            file_groups[base_name][grab_num] = file

    # Process each group
    for base_name, grabs in file_groups.items():
        if '0' in grabs and '1' in grabs:
            print(f"Processing {base_name}...")

            # Read both CSV files
            df0 = pd.read_csv(os.path.join(folder_path, grabs['0']))
            df1 = pd.read_csv(os.path.join(folder_path, grabs['1']))

            # Ensure both dataframes have same number of rows
            if len(df0) != len(df1):
                print(f"Warning: {grabs['0']} and {grabs['1']} have different row counts, using smaller count")
                min_rows = min(len(df0), len(df1))
                df0 = df0.head(min_rows)
                df1 = df1.head(min_rows)

            # Prepare result dataframe
            result_df = pd.DataFrame()
            result_df['frame'] = df0['frame']

            # Store 3D coordinates
            cx_list, cy_list, cz_list = [], [], []

            # Process each frame
            for idx in range(len(df0)):
                try:
                    # Extract parameters
                    OID = df0.iloc[idx]['OID']
                    CouchAngle = df0.iloc[idx]['CouchAngle']
                    IPEL = df0.iloc[idx]['IPEL']

                    # Extract X-ray source coordinates
                    XSA = [
                        df0.iloc[idx]['X tube x'],
                        df0.iloc[idx]['X tube y'],
                        df0.iloc[idx]['X tube z']
                    ]

                    XSB = [
                        df1.iloc[idx]['X tube x'],
                        df1.iloc[idx]['X tube y'],
                        df1.iloc[idx]['X tube z']
                    ]

                    # Extract marker point coordinates
                    PA = (df0.iloc[idx]['marker_x'], df0.iloc[idx]['marker_y'])
                    PB = (df1.iloc[idx]['marker_x'], df1.iloc[idx]['marker_y'])

                    # Calculate intersection
                    C = compute_intersection(XSA, XSB, PA, PB, OID, CouchAngle, resolution, IPEL)

                    cx_list.append(C[0])
                    cy_list.append(C[1])
                    cz_list.append(C[2])

                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    cx_list.append(np.nan)
                    cy_list.append(np.nan)
                    cz_list.append(np.nan)

            # Add results to dataframe
            result_df['Cx'] = cx_list
            result_df['Cy'] = cy_list
            result_df['Cz'] = cz_list

            # Save results
            output_path = os.path.join(folder_path, f"{base_name}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    folder_path = "path/to/trajectory/marker/data"
    resolution = 256  # Image resolution, adjust according to actual situation

    try:
        process_marker_files(folder_path, resolution)
        print("Batch processing completed")
    except Exception as e:
        print(f"Error during processing: {e}")

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')


def butter_lowpass(cutoff, fs, order=4):
    """Design Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=4):
    """Apply lowpass filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def analyze_global_frequency(markers_folder, sampling_rate=30, output_dir=None):
    """
    Analyze frequency characteristics of all marker trajectories to determine optimal cutoff frequency.
    
    Parameters:
    markers_folder: Path to folder containing marker CSV files
    sampling_rate: Sampling rate (frames per second)
    output_dir: Directory for output visualizations
    
    Returns:
    Recommended global cutoff frequency based on frequency distribution gap
    """
    marker_files = [f for f in os.listdir(markers_folder) if f.endswith('.csv')]

    all_frequencies = []
    all_amplitudes = []

    for file in marker_files:
        try:
            marker_df = pd.read_csv(os.path.join(markers_folder, file))

            if 'marker_x' in marker_df.columns and 'marker_y' in marker_df.columns:
                # Analyze X coordinate frequency
                x_data = marker_df['marker_x'].values
                x_data = x_data - np.mean(x_data)  # Remove mean

                n = len(x_data)
                yf = fft(x_data)
                xf = fftfreq(n, 1 / sampling_rate)

                idx = np.where(xf > 0)
                xf = xf[idx]
                yf = np.abs(yf[idx])

                for freq, amp in zip(xf, yf):
                    if freq > 0.1:  # Ignore frequencies below 0.1Hz
                        all_frequencies.append(freq)
                        all_amplitudes.append(amp)

                # Analyze Y coordinate frequency
                y_data = marker_df['marker_y'].values
                y_data = y_data - np.mean(y_data)  # Remove mean

                n = len(y_data)
                yf = fft(y_data)
                xf = fftfreq(n, 1 / sampling_rate)

                idx = np.where(xf > 0)
                xf = xf[idx]
                yf = np.abs(yf[idx])

                for freq, amp in zip(xf, yf):
                    if freq > 0.1:  # Ignore frequencies below 0.1Hz
                        all_frequencies.append(freq)
                        all_amplitudes.append(amp)

        except Exception as e:
            print(f"Error analyzing frequency for file {file}: {e}")
            continue

    if not all_frequencies:
        print("Warning: No valid frequency data found, using default cutoff frequency 0.7Hz")
        return 0.7

    all_frequencies = np.array(all_frequencies)
    all_amplitudes = np.array(all_amplitudes)

    # Create frequency distribution visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Frequency-amplitude scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(all_frequencies, all_amplitudes, alpha=0.5, s=10, color='#2a9d8f')
        plt.xlabel('Frequency (Hz)', fontsize=24)
        plt.ylabel('Amplitude', fontsize=24)
        plt.title('Frequency-Amplitude Distribution of All Marker Tracks', fontsize=24)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(os.path.join(output_dir, 'frequency_amplitude_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Frequency histogram
        plt.figure(figsize=(12, 8))
        plt.hist(all_frequencies, bins=100, color='#2a9d8f', alpha=0.7, edgecolor='black')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Frequency Count')
        plt.title('Frequency Distribution Histogram of All Marker Tracks')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'frequency_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Cumulative frequency distribution
        sorted_freqs = np.sort(all_frequencies)
        cum_freq = np.arange(1, len(sorted_freqs) + 1) / len(sorted_freqs)

        plt.figure(figsize=(12, 8))
        plt.plot(sorted_freqs, cum_freq, color='#2a9d8f', linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Cumulative Frequency')
        plt.title('Cumulative Frequency Distribution of All Marker Tracks')
        plt.grid(True, alpha=0.3)

        # Mark 90th and 95th percentiles
        cutoff_90 = np.percentile(all_frequencies, 90)
        cutoff_95 = np.percentile(all_frequencies, 95)
        
        plt.axvline(x=cutoff_90, color='#e76f51', linestyle='--', linewidth=2,
                    label=f'90th percentile: {cutoff_90:.2f} Hz')
        plt.axhline(y=0.9, color='#e76f51', linestyle='--', linewidth=2)
        
        plt.axvline(x=cutoff_95, color='#f4a261', linestyle='--', linewidth=2,
                    label=f'95th percentile: {cutoff_95:.2f} Hz')
        plt.axhline(y=0.95, color='#f4a261', linestyle='--', linewidth=2)

        plt.legend()
        plt.savefig(os.path.join(output_dir, 'cumulative_frequency.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Frequency-amplitude density plot
        plt.figure(figsize=(12, 8))
        hb = plt.hexbin(all_frequencies, all_amplitudes, gridsize=50, cmap='viridis', bins='log')
        plt.colorbar(hb, label='Log Count')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Frequency-Amplitude Density Plot (Gap Detection)')
        plt.yscale('log')
        plt.savefig(os.path.join(output_dir, 'frequency_amplitude_density.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Return 95th percentile as recommended cutoff
    return np.percentile(all_frequencies, 95)


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
    D_vec = O - S
    dist_SO = np.linalg.norm(D_vec)
    if dist_SO < 1e-10:
        raise ValueError("X-ray source located at origin, cannot define direction.")
    D_unit = D_vec / dist_SO

    C_plane = O + OID * D_unit

    dx, dy, dz = D_unit
    L_xy = np.sqrt(dx * dx + dy * dy)
    if L_xy < 1e-10:
        X_vec = np.array([1.0, 0.0, 0.0])
    else:
        X_vec = np.array([dy, -dx, 0.0]) / L_xy

    Y_vec = np.cross(D_unit, X_vec)

    pixel_size = IPEL / resolution
    px_mm, py_mm = (P_img[0] - resolution / 2) * pixel_size, (P_img[1] - resolution / 2) * pixel_size

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
    XSA_rotated = rotate_around_z(XSA[0], XSA[1], XSA[2], -CouchAngle)
    XSB_rotated = rotate_around_z(XSB[0], XSB[1], XSB[2], -CouchAngle)

    IPA = compute_image_point(XSA_rotated, PA, OID, resolution, IPEL)
    IPB = compute_image_point(XSB_rotated, PB, OID, resolution, IPEL)

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

    if abs(1 - d * d) < 1e-10:
        if np.allclose(u_unit, v_unit):
            proj = np.dot(B0 - A0, u_unit)
            if proj >= 0:
                P = A0 + proj * u_unit
                Q = B0
            else:
                P = A0
                Q = B0 + (-proj) * u_unit
        elif np.allclose(u_unit, -v_unit):
            lambda0 = np.dot(B0 - A0, u_unit)
            if lambda0 >= 0:
                P = A0 + lambda0 * u_unit
                Q = B0
            else:
                P = A0
                Q = B0
        else:
            P = A0
            Q = B0
    else:
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


def process_tumor_data(markers_folder, tumor_folder, resolution, sampling_rate=30, global_cutoff_freq=8):
    """
    Process tumor center data with adaptive low-pass filtering and calculate 3D trajectory points.
    
    Parameters:
    markers_folder: Path to folder containing marker CSV files
    tumor_folder: Path to folder containing tumor center CSV files
    resolution: Image resolution in pixels
    sampling_rate: Sampling rate (frames per second)
    global_cutoff_freq: Global cutoff frequency for low-pass filtering
    
    Returns:
    None, generates new CSV files in tumor folder
    """
    marker_files = [f for f in os.listdir(markers_folder) if f.endswith('.csv')]

    marker_groups = {}
    for file in marker_files:
        match = re.match(r'(PT\d+_F\d+)_grab(\d+)\.csv', file)
        if match:
            base_name = match.group(1)
            grab_num = match.group(2)

            if base_name not in marker_groups:
                marker_groups[base_name] = {}

            marker_groups[base_name][grab_num] = file

    for base_name, grabs in marker_groups.items():
        if '0' in grabs and '1' in grabs:
            print(f"Processing {base_name}...")

            marker_df0 = pd.read_csv(os.path.join(markers_folder, grabs['0']))
            marker_df1 = pd.read_csv(os.path.join(markers_folder, grabs['1']))

            tumor_file0 = os.path.join(tumor_folder, grabs['0'])
            tumor_file1 = os.path.join(tumor_folder, grabs['1'])

            if not (os.path.exists(tumor_file0) and os.path.exists(tumor_file1)):
                print(f"Warning: {grabs['0']} or {grabs['1']} not found in tumor folder, skipping")
                continue

            tumor_df0 = pd.read_csv(tumor_file0)
            tumor_df1 = pd.read_csv(tumor_file1)

            min_rows = min(len(marker_df0), len(marker_df1), len(tumor_df0), len(tumor_df1))
            marker_df0 = marker_df0.head(min_rows)
            marker_df1 = marker_df1.head(min_rows)
            tumor_df0 = tumor_df0.head(min_rows)
            tumor_df1 = tumor_df1.head(min_rows)

            # Calculate alignment offsets
            MF_x0, MF_y0 = marker_df0['marker_x'].mean(), marker_df0['marker_y'].mean()
            MF_x1, MF_y1 = marker_df1['marker_x'].mean(), marker_df1['marker_y'].mean()
            MT_x0, MT_y0 = tumor_df0['tumor_x'].mean(), tumor_df0['tumor_y'].mean()
            MT_x1, MT_y1 = tumor_df1['tumor_x'].mean(), tumor_df1['tumor_y'].mean()

            offset_x0, offset_y0 = MT_x0 - MF_x0, MT_y0 - MF_y0
            offset_x1, offset_y1 = MT_x1 - MF_x1, MT_y1 - MF_y1

            # Align tumor coordinates
            aligned_x0 = tumor_df0['tumor_x'] - offset_x0
            aligned_y0 = tumor_df0['tumor_y'] - offset_y0
            aligned_x1 = tumor_df1['tumor_x'] - offset_x1
            aligned_y1 = tumor_df1['tumor_y'] - offset_y1

            # Apply low-pass filtering
            filtered_x0 = lowpass_filter(aligned_x0, global_cutoff_freq, sampling_rate)
            filtered_y0 = lowpass_filter(aligned_y0, global_cutoff_freq, sampling_rate)
            filtered_x1 = lowpass_filter(aligned_x1, global_cutoff_freq, sampling_rate)
            filtered_y1 = lowpass_filter(aligned_y1, global_cutoff_freq, sampling_rate)

            result_df = pd.DataFrame()
            result_df['frame'] = marker_df0['frame']

            cx_list, cy_list, cz_list = [], [], []

            for idx in range(min_rows):
                try:
                    OID = marker_df0.iloc[idx]['OID']
                    CouchAngle = marker_df0.iloc[idx]['CouchAngle']
                    IPEL = marker_df0.iloc[idx]['IPEL']

                    XSA = [
                        marker_df0.iloc[idx]['X tube x'],
                        marker_df0.iloc[idx]['X tube y'],
                        marker_df0.iloc[idx]['X tube z']
                    ]

                    XSB = [
                        marker_df1.iloc[idx]['X tube x'],
                        marker_df1.iloc[idx]['X tube y'],
                        marker_df1.iloc[idx]['X tube z']
                    ]

                    PA = (filtered_x0[idx], filtered_y0[idx])
                    PB = (filtered_x1[idx], filtered_y1[idx])

                    C = compute_intersection(XSA, XSB, PA, PB, OID, CouchAngle, resolution, IPEL)

                    cx_list.append(C[0])
                    cy_list.append(C[1])
                    cz_list.append(C[2])

                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    cx_list.append(np.nan)
                    cy_list.append(np.nan)
                    cz_list.append(np.nan)

            result_df['Cx'] = cx_list
            result_df['Cy'] = cy_list
            result_df['Cz'] = cz_list

            output_dir = os.path.join(tumor_folder, "3DTrajectories_LPF_adaptive")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{base_name}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"Tumor trajectory results saved to {output_path}")


if __name__ == "__main__":
    markers_folder = "path/to/marker/data"
    tumor_folder = "path/to/tumor/center/data"
    resolution = 256
    sampling_rate = 30
    
    # Optional: Analyze global frequency characteristics
    # output_dir = os.path.join(markers_folder, "FrequencyAnalysis")
    # global_cutoff = analyze_global_frequency(markers_folder, sampling_rate, output_dir)
    
    try:
        process_tumor_data(markers_folder, tumor_folder, resolution, sampling_rate)
        print("Tumor data processing completed")
    except Exception as e:
        print(f"Error during processing: {e}")

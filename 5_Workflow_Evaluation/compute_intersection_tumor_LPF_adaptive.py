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


# 低通滤波器设计
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def analyze_global_frequency(markers_folder, sampling_rate=30):
    """
    分析所有marker轨迹的频率特性，找到最大频率

    参数:
    markers_folder: 包含基准标记CSV文件的文件夹路径
    sampling_rate: 采样率（帧/秒）

    返回:
    建议的全局截止频率（最大频率的1.5倍）
    """
    # 获取基准标记文件夹中的所有CSV文件
    marker_files = [f for f in os.listdir(markers_folder) if f.endswith('.csv')]

    all_max_frequencies = []

    for file in marker_files:
        try:
            # 读取marker数据
            marker_df = pd.read_csv(os.path.join(markers_folder, file))

            # 分析每个marker文件的x和y坐标
            if 'marker_x' in marker_df.columns and 'marker_y' in marker_df.columns:
                # 分析X坐标的频率
                x_data = marker_df['marker_x'].values
                x_data = x_data - np.mean(x_data)  # 去除均值

                # 计算FFT
                n = len(x_data)
                yf = fft(x_data)
                xf = fftfreq(n, 1 / sampling_rate)

                # 只取正频率部分
                idx = np.where(xf > 0)
                xf = xf[idx]
                yf = np.abs(yf[idx])

                # 找到有显著能量的频率（排除噪声）
                energy_threshold = np.max(yf) * 0.1  # 能量阈值为最大能量的10%
                significant_freqs = xf[yf > energy_threshold]
                if len(significant_freqs) > 0:
                    max_freq_x = np.max(significant_freqs)
                    all_max_frequencies.append(max_freq_x)

                # 分析Y坐标的频率
                y_data = marker_df['marker_y'].values
                y_data = y_data - np.mean(y_data)  # 去除均值

                # 计算FFT
                n = len(y_data)
                yf = fft(y_data)
                xf = fftfreq(n, 1 / sampling_rate)

                # 只取正频率部分
                idx = np.where(xf > 0)
                xf = xf[idx]
                yf = np.abs(yf[idx])

                # 找到有显著能量的频率（排除噪声）
                energy_threshold = np.max(yf) * 0.1  # 能量阈值为最大能量的10%
                significant_freqs = xf[yf > energy_threshold]
                if len(significant_freqs) > 0:
                    max_freq_y = np.max(significant_freqs)
                    all_max_frequencies.append(max_freq_y)

        except Exception as e:
            print(f"分析文件 {file} 的频率时出错: {e}")
            continue

    if not all_max_frequencies:
        print("警告: 没有找到有效的频率数据，使用默认截止频率0.7Hz")
        return 0.7

    # 找到所有频率中的最大值
    global_max_freq = np.max(all_max_frequencies)

    cutoff_freq = np.percentile(all_max_frequencies, 90)

    print(f"所有marker轨迹中90%的最大频率都低于: {global_max_freq:.2f} Hz")
    print(f"确定的全局截止频率: {cutoff_freq:.2f} Hz")

    return cutoff_freq


def rotate_around_z(x, y, z, angle_degrees):
    """
    绕Z轴旋转点坐标

    参数:
    x, y, z: 点的坐标
    angle_degrees: 旋转角度（度）

    返回:
    旋转后的坐标数组
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
    计算成像平面上的三维点坐标。

    参数:
    S: 三维数组，X射线源点坐标
    P_img: 二维元组，成像平面上的点坐标(px, py)，单位为像素
    OID: 标量，原点到成像平面的距离（mm）
    resolution: 图像分辨率（像素）
    IPEL: 成像平面边长（mm）

    返回:
    三维数组，成像点坐标
    """
    O = np.array([0.0, 0.0, 0.0])
    D_vec = O - S  # 向量从S到O
    dist_SO = np.linalg.norm(D_vec)
    if dist_SO < 1e-10:
        raise ValueError("X射线源位于原点，无法定义方向。")
    D_unit = D_vec / dist_SO

    # 计算成像平面中心点
    C_plane = O + OID * D_unit

    # 计算X轴向量
    dx, dy, dz = D_unit
    L_xy = np.sqrt(dx * dx + dy * dy)
    if L_xy < 1e-10:
        # 如果D_unit的X和Y分量为零，则X轴取(1,0,0)
        X_vec = np.array([1.0, 0.0, 0.0])
    else:
        X_vec = np.array([dy, -dx, 0.0]) / L_xy

    # 计算Y轴向量: Y_vec = - (D_unit × X_vec)
    Y_vec = np.cross(D_unit, X_vec)

    # 将像素坐标转换为毫米坐标
    pixel_size = IPEL / resolution  # 每个像素的大小（mm）
    px_mm, py_mm = (P_img[0] - resolution / 2) * pixel_size, (P_img[1] - resolution / 2) * pixel_size

    # 计算成像点三维坐标
    point_3d = C_plane + px_mm * X_vec + py_mm * Y_vec
    return point_3d


def compute_intersection(XSA, XSB, PA, PB, OID, CouchAngle, resolution, IPEL):
    """
    计算两条X射线的交点或最接近点中点C。

    参数:
    XSA: 三维数组，第一个X射线源点坐标（mm）
    XSB: 三维数组，第二个X射线源点坐标（mm）
    PA: 二维元组，第一个成像平面上的点坐标(px, py)，单位为像素
    PB: 二维元组，第二个成像平面上的点坐标(px, py)，单位为像素
    OID: 标量，原点到成像平面的距离（mm）
    CouchAngle: 治疗床旋转角度（度）
    resolution: 图像分辨率（像素）
    IPEL: 成像平面边长（mm）

    返回:
    三维数组，交点C的坐标
    """
    # 旋转X射线源点
    XSA_rotated = rotate_around_z(XSA[0], XSA[1], XSA[2], -CouchAngle)
    XSB_rotated = rotate_around_z(XSB[0], XSB[1], XSB[2], -CouchAngle)

    # 计算成像平面上的三维点IPA和IPB
    IPA = compute_image_point(XSA_rotated, PA, OID, resolution, IPEL)
    IPB = compute_image_point(XSB_rotated, PB, OID, resolution, IPEL)

    # 计算射线方向向量并归一化
    u_dir = IPA - XSA_rotated
    v_dir = IPB - XSB_rotated
    u_norm = np.linalg.norm(u_dir)
    v_norm = np.linalg.norm(v_dir)
    if u_norm < 1e-10 or v_norm < 1e-10:
        raise ValueError("射线长度为零，无法计算。")
    u_unit = u_dir / u_norm
    v_unit = v_dir / v_norm

    A0 = XSA_rotated
    B0 = XSB_rotated
    W0 = A0 - B0

    d = np.dot(u_unit, v_unit)

    # 检查射线是否平行
    if abs(1 - d * d) < 1e-10:
        # 处理平行射线
        if np.allclose(u_unit, v_unit):
            # 同向
            proj = np.dot(B0 - A0, u_unit)
            if proj >= 0:
                P = A0 + proj * u_unit
                Q = B0
            else:
                P = A0
                Q = B0 + (-proj) * u_unit
        elif np.allclose(u_unit, -v_unit):
            # 反向
            lambda0 = np.dot(B0 - A0, u_unit)
            if lambda0 >= 0:
                P = A0 + lambda0 * u_unit
                Q = B0
            else:
                P = A0
                Q = B0
        else:
            # 其他情况（理论上不应发生），取源点
            P = A0
            Q = B0
    else:
        # 处理不平行射线
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


def process_tumor_data(markers_folder, tumor_folder, resolution, sampling_rate=30):
    """
    处理肿瘤中心数据，与基准标记对齐后计算三维轨迹点

    参数:
    markers_folder: 包含基准标记CSV文件的文件夹路径
    tumor_folder: 包含肿瘤中心CSV文件的文件夹路径
    resolution: 图像分辨率（像素）
    sampling_rate: 采样率（帧/秒）

    返回:
    无，但会在肿瘤文件夹中生成新的CSV文件
    """
    # 首先分析所有marker轨迹，确定全局截止频率
    global_cutoff_freq = analyze_global_frequency(markers_folder, sampling_rate)
    print(f"使用全局截止频率: {global_cutoff_freq:.2f} Hz")

    # 获取基准标记文件夹中的所有CSV文件
    marker_files = [f for f in os.listdir(markers_folder) if f.endswith('.csv')]

    # 按患者和治疗野分组文件
    marker_groups = {}
    for file in marker_files:
        match = re.match(r'(PT\d+_F\d+)_grab(\d+)\.csv', file)
        if match:
            base_name = match.group(1)
            grab_num = match.group(2)

            if base_name not in marker_groups:
                marker_groups[base_name] = {}

            marker_groups[base_name][grab_num] = file

    # 处理每个组
    for base_name, grabs in marker_groups.items():
        if '0' in grabs and '1' in grabs:
            print(f"处理 {base_name}...")

            # 读取基准标记CSV文件
            marker_df0 = pd.read_csv(os.path.join(markers_folder, grabs['0']))
            marker_df1 = pd.read_csv(os.path.join(markers_folder, grabs['1']))

            # 检查肿瘤文件夹中是否有对应的文件
            tumor_file0 = os.path.join(tumor_folder, grabs['0'])
            tumor_file1 = os.path.join(tumor_folder, grabs['1'])

            if not (os.path.exists(tumor_file0) and os.path.exists(tumor_file1)):
                print(f"警告: 肿瘤文件夹中找不到 {grabs['0']} 或 {grabs['1']}，跳过")
                continue

            # 读取肿瘤中心CSV文件
            tumor_df0 = pd.read_csv(tumor_file0)
            tumor_df1 = pd.read_csv(tumor_file1)

            # 确保所有数据帧的行数相同
            min_rows = min(len(marker_df0), len(marker_df1), len(tumor_df0), len(tumor_df1))
            marker_df0 = marker_df0.head(min_rows)
            marker_df1 = marker_df1.head(min_rows)
            tumor_df0 = tumor_df0.head(min_rows)
            tumor_df1 = tumor_df1.head(min_rows)

            # 计算基准标记的平均坐标
            MF_x0 = marker_df0['marker_x'].mean()
            MF_y0 = marker_df0['marker_y'].mean()
            MF_x1 = marker_df1['marker_x'].mean()
            MF_y1 = marker_df1['marker_y'].mean()

            # 计算肿瘤中心的平均坐标
            MT_x0 = tumor_df0['tumor_x'].mean()
            MT_y0 = tumor_df0['tumor_y'].mean()
            MT_x1 = tumor_df1['tumor_x'].mean()
            MT_y1 = tumor_df1['tumor_y'].mean()

            # 计算对齐偏移量
            offset_x0 = MT_x0 - MF_x0
            offset_y0 = MT_y0 - MF_y0
            offset_x1 = MT_x1 - MF_x1
            offset_y1 = MT_y1 - MF_y1

            # 提取对齐后的肿瘤坐标
            aligned_x0 = tumor_df0['tumor_x'] - offset_x0
            aligned_y0 = tumor_df0['tumor_y'] - offset_y0
            aligned_x1 = tumor_df1['tumor_x'] - offset_x1
            aligned_y1 = tumor_df1['tumor_y'] - offset_y1

            # 应用低通滤波器（使用全局截止频率）
            filtered_x0 = lowpass_filter(aligned_x0, global_cutoff_freq, sampling_rate)
            filtered_y0 = lowpass_filter(aligned_y0, global_cutoff_freq, sampling_rate)
            filtered_x1 = lowpass_filter(aligned_x1, global_cutoff_freq, sampling_rate)
            filtered_y1 = lowpass_filter(aligned_y1, global_cutoff_freq, sampling_rate)

            # 准备结果数据帧
            result_df = pd.DataFrame()
            result_df['frame'] = marker_df0['frame']

            # 存储三维坐标
            cx_list, cy_list, cz_list = [], [], []

            # 处理每一帧
            for idx in range(min_rows):
                try:
                    # 提取参数
                    OID = marker_df0.iloc[idx]['OID']
                    CouchAngle = marker_df0.iloc[idx]['CouchAngle']
                    IPEL = marker_df0.iloc[idx]['IPEL']

                    # 提取X射线源坐标
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

                    # 使用滤波后的肿瘤中心点坐标
                    filtered_x0 = pd.DataFrame(filtered_x0)
                    filtered_y0 = pd.DataFrame(filtered_y0)
                    filtered_x1 = pd.DataFrame(filtered_x1)
                    filtered_y1 = pd.DataFrame(filtered_y1)
                    PA = (filtered_x0.iloc[idx].values, filtered_y0.iloc[idx].values)
                    PB = (filtered_x1.iloc[idx].values, filtered_y1.iloc[idx].values)

                    # 计算交点
                    C = compute_intersection(XSA, XSB, PA, PB, OID, CouchAngle, resolution, IPEL)

                    cx_list.append(C[0])
                    cy_list.append(C[1])
                    cz_list.append(C[2])

                except Exception as e:
                    print(f"处理第 {idx} 帧时出错: {e}")
                    cx_list.append(np.nan)
                    cy_list.append(np.nan)
                    cz_list.append(np.nan)

            # 添加结果到数据帧
            result_df['Cx'] = cx_list
            result_df['Cy'] = cy_list
            result_df['Cz'] = cz_list

            # 创建输出目录
            output_dir = os.path.join(tumor_folder, "3DTrajectories_LPF_adaptive")
            os.makedirs(output_dir, exist_ok=True)

            # 保存结果
            output_path = os.path.join(output_dir, f"{base_name}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"已保存肿瘤轨迹结果到 {output_path}")


# 示例用法
if __name__ == "__main__":
    markers_folder = "C:/D/PatientData/Trajectories/Markers"
    tumor_folder = "C:/D/PatientData/Trajectories/TumorCenters-Alpha0_07"
    resolution = 256  # 图像分辨率，需要根据实际情况设置
    sampling_rate = 30  # 采样率（帧/秒）

    try:
        process_tumor_data(markers_folder, tumor_folder, resolution, sampling_rate)
        print("肿瘤数据处理完成")
    except Exception as e:
        print(f"处理过程中出错: {e}")
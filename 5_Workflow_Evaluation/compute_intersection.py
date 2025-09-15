import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    '''
        # 创建3D图形 可视化验证
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制X射线源点
        ax.scatter(*A0, color='red', s=100, label='XSA')
        ax.scatter(*B0, color='blue', s=100, label='XSB')

        # 绘制成像点
        ax.scatter(*IPA, color='red', s=50, marker='^', label='IPA')
        ax.scatter(*IPB, color='blue', s=50, marker='^', label='IPB')

        # 绘制P和Q点
        ax.scatter(*P, color='green', s=100, marker='o', label='P (Closest on Ray A)')
        ax.scatter(*Q, color='purple', s=100, marker='o', label='Q (Closest on Ray B)')

        # 绘制中点C
        ax.scatter(*C, color='orange', s=150, marker='X', label='C (Midpoint)')

        # 绘制射线
        # 射线A: 从XSA到IPA，并延长一些
        ray_a_length = np.linalg.norm(IPA - A0)
        ray_a_extended = A0 + 1.5 * ray_a_length * u_unit
        ax.plot([A0[0], ray_a_extended[0]], 
                [A0[1], ray_a_extended[1]], 
                [A0[2], ray_a_extended[2]], 'r-', linewidth=2, alpha=0.7)

        # 射线B: 从XSB到IPB，并延长一些
        ray_b_length = np.linalg.norm(IPB - B0)
        ray_b_extended = B0 + 1.5 * ray_b_length * v_unit
        ax.plot([B0[0], ray_b_extended[0]], 
                [B0[1], ray_b_extended[1]], 
                [B0[2], ray_b_extended[2]], 'b-', linewidth=2, alpha=0.7)

        # 绘制P和Q之间的连线
        ax.plot([P[0], Q[0]], 
                [P[1], Q[1]], 
                [P[2], Q[2]], 'g--', linewidth=2, alpha=0.7, label='PQ (Shortest Distance)')

        # 设置坐标轴标签
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        # 设置标题
        ax.set_title('X-Ray Intersection Visualization')

        # 添加图例
        ax.legend()

        # 添加网格
        ax.grid(True)

        # 显示图形
        plt.show()

        # 打印点坐标信息
        print(f"XSA: {A0}")
        print(f"XSB: {B0}")
        print(f"IPA: {IPA}")
        print(f"IPB: {IPB}")
        print(f"P (Closest on Ray A): {P}")
        print(f"Q (Closest on Ray B): {Q}")
        print(f"C (Midpoint): {C}")
        print(f"Distance between P and Q: {np.linalg.norm(P - Q):.4f} mm")
        '''

    return C


def process_marker_files(folder_path, resolution):
    """
    批量处理标记文件，计算三维轨迹点

    参数:
    folder_path: 包含CSV文件的文件夹路径
    resolution: 图像分辨率（像素）

    返回:
    无，但会在同一文件夹中生成新的CSV文件
    """
    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 按患者和治疗野分组文件
    file_groups = {}
    for file in csv_files:
        # 使用正则表达式提取PT和F信息
        match = re.match(r'(PT\d+_F\d+)_grab(\d+)\.csv', file)
        if match:
            base_name = match.group(1)
            grab_num = match.group(2)

            if base_name not in file_groups:
                file_groups[base_name] = {}

            file_groups[base_name][grab_num] = file

    # 处理每个组
    for base_name, grabs in file_groups.items():
        if '0' in grabs and '1' in grabs:
            print(f"处理 {base_name}...")

            # 读取两个CSV文件
            df0 = pd.read_csv(os.path.join(folder_path, grabs['0']))
            df1 = pd.read_csv(os.path.join(folder_path, grabs['1']))

            # 确保两个数据帧的行数相同
            if len(df0) != len(df1):
                print(f"警告: {grabs['0']} 和 {grabs['1']} 的行数不同，将以较小的行数为准")
                min_rows = min(len(df0), len(df1))
                df0 = df0.head(min_rows)
                df1 = df1.head(min_rows)

            # 准备结果数据帧
            result_df = pd.DataFrame()
            result_df['frame'] = df0['frame']

            # 存储三维坐标
            cx_list, cy_list, cz_list = [], [], []

            # 处理每一帧
            for idx in range(len(df0)):
                try:
                    # 提取参数
                    OID = df0.iloc[idx]['OID']
                    CouchAngle = df0.iloc[idx]['CouchAngle']
                    IPEL = df0.iloc[idx]['IPEL']

                    # 提取X射线源坐标
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

                    # 提取标记点坐标
                    PA = (df0.iloc[idx]['marker_x'], df0.iloc[idx]['marker_y'])
                    PB = (df1.iloc[idx]['marker_x'], df1.iloc[idx]['marker_y'])

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

            # 保存结果
            output_path = os.path.join(folder_path, f"{base_name}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"已保存结果到 {output_path}")


# 示例用法
if __name__ == "__main__":
    folder_path = "C:/D/PatientData/Trajectories/Markers"
    resolution = 256  # 图像分辨率，需要根据实际情况设置

    try:
        process_marker_files(folder_path, resolution)
        print("批量处理完成")
    except Exception as e:
        print(f"处理过程中出错: {e}")
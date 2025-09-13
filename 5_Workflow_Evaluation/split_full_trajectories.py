import os
import pandas as pd
import ast
import re


def split_full_trajectories():
    # 文件路径
    full_trajectories_path = "C:/D/PatientData/Trajectories/TumorCenters/full_trajectories.csv"
    markers_folder = "C:/D/PatientData/Trajectories/Markers"
    output_folder = "C:/D/PatientData/Trajectories/TumorCenters"

    # 读取完整轨迹文件
    df_full = pd.read_csv(full_trajectories_path)

    # 处理每一行数据
    for index, row in df_full.iterrows():
        # 获取patient, field, device值
        patient = str(row['patient'])
        field = str(row['field'])
        device = str(row['device'])

        # 创建文件名
        filename = f"{patient}_{field}_{device}.csv"
        output_path = os.path.join(output_folder, filename)

        # 解析tumor_traj列
        try:
            # 尝试解析字符串为列表
            tumor_traj = ast.literal_eval(row['tumor_traj'])
        except:
            # 如果解析失败，尝试使用正则表达式提取坐标
            coords = re.findall(r'\((\d+),\s*(\d+)\)', row['tumor_traj'])
            tumor_traj = [(int(x), int(y)) for x, y in coords]

        # 创建新的DataFrame
        new_data = []
        for frame, (x, y) in enumerate(tumor_traj):
            new_data.append({
                'frame': frame,
                'tumor_x': x,
                'tumor_y': y
            })

        df_new = pd.DataFrame(new_data)

        # 从Markers文件夹中获取对应的CSV文件
        marker_file_path = os.path.join(markers_folder, filename)

        if os.path.exists(marker_file_path):
            # 读取对应的marker文件
            df_marker = pd.read_csv(marker_file_path)

            # 获取需要的列数据（取第一行的值）
            if not df_marker.empty:
                if 'X tube x' in df_marker.columns:
                    x_tube_x = df_marker.iloc[0]['X tube x']
                    x_tube_y = df_marker.iloc[0]['X tube y']
                    x_tube_z = df_marker.iloc[0]['X tube z']
                    ipel = df_marker.iloc[0]['IPEL'] if 'IPEL' in df_marker.columns else 233.38
                    oid = df_marker.iloc[0]['OID'] if 'OID' in df_marker.columns else 2444
                    couch_angle = df_marker.iloc[0]['CouchAngle'] if 'CouchAngle' in df_marker.columns else 0

                    # 添加这些列到新DataFrame
                    df_new['X tube x'] = x_tube_x
                    df_new['X tube y'] = x_tube_y
                    df_new['X tube z'] = x_tube_z
                    df_new['IPEL'] = ipel
                    df_new['OID'] = oid
                    df_new['CouchAngle'] = couch_angle
                else:
                    print(f"警告: {filename} 中缺少必要的列，使用默认值")
                    # 使用默认值
                    df_new['X tube x'] = 0
                    df_new['X tube y'] = 0
                    df_new['X tube z'] = 0
                    df_new['IPEL'] = 233.38
                    df_new['OID'] = 2444
                    df_new['CouchAngle'] = 0
            else:
                print(f"警告: {filename} 为空文件，使用默认值")
                # 使用默认值
                df_new['X tube x'] = 0
                df_new['X tube y'] = 0
                df_new['X tube z'] = 0
                df_new['IPEL'] = 233.38
                df_new['OID'] = 2444
                df_new['CouchAngle'] = 0
        else:
            print(f"警告: 找不到对应的marker文件 {filename}，使用默认值")
            # 使用默认值
            df_new['X tube x'] = 0
            df_new['X tube y'] = 0
            df_new['X tube z'] = 0
            df_new['IPEL'] = 233.38
            df_new['OID'] = 2444
            df_new['CouchAngle'] = 0

        # 确保列的顺序正确
        column_order = ['frame', 'tumor_x', 'tumor_y', 'X tube x', 'X tube y', 'X tube z', 'IPEL', 'OID', 'CouchAngle']
        df_new = df_new[column_order]

        # 保存到CSV文件
        df_new.to_csv(output_path, index=False)
        print(f"已保存: {filename} (共{len(tumor_traj)}帧)")

    print("所有文件处理完成!")


# 运行函数
if __name__ == "__main__":
    split_full_trajectories()
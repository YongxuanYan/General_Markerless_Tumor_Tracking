import os
import pandas as pd


machine_coordinates = {
    '1': [-1332, 844, -1504],
    '2': [1332, 844, -1504],
    '3': [1332, -844, -1504],
    '4': [-1332, -844, -1504]
}
folder_path = "C:/D/PatientData/Trajectories"


def process_csv_files(folder_path):
    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 1. 删除指定列（如果存在）
        columns_to_drop = ['patient', 'field', 'device']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # 2. 询问用户机器号
        while True:
            machine_num = input(f"请问 {csv_file} 是几号机器？(请输入1, 2, 3或4): ").strip()
            if machine_num in ['1', '2', '3', '4']:
                break
            else:
                print("输入无效，请重新输入1, 2, 3或4")

        # 获取对应的X tube坐标
        x_tube_coords = machine_coordinates[machine_num]

        # 添加新列
        df['X tube x'] = x_tube_coords[0]
        df['X tube y'] = x_tube_coords[1]
        df['X tube z'] = x_tube_coords[2]
        df['IPEL'] = 233.38
        df['OID'] = 2444

        # 保存处理后的文件并覆盖源文件
        filepath = os.path.join(folder_path, csv_file)
        df.to_csv(filepath, index=False)

        print(f"已处理并保存: {csv_file}")


process_csv_files(folder_path)

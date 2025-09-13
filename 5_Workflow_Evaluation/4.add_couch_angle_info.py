import os
import pandas as pd


def add_couch_angle_info(folder_path):
    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 询问用户治疗床角度
        while True:
            try:
                couch_angle = float(input(f"请问 {csv_file} 的治疗床角度是多少度？(范围为[0~360)): ").strip())
                if 0 <= couch_angle < 360:
                    break
                else:
                    print("输入无效，请输入0到360之间的数值（不包括360）")
            except ValueError:
                print("输入无效，请输入有效的数字")

        # 添加CouchAngle列
        df['CouchAngle'] = couch_angle

        # 保存处理后的文件（可以选择覆盖原文件或保存为新文件）
        # 这里选择在原文件名前加"with_angle_"作为新文件名
        filepath = os.path.join(folder_path, csv_file)
        df.to_csv(filepath, index=False)

        print(f"已添加治疗床角度并保存: {csv_file}")


# 使用示例
if __name__ == "__main__":
    folder_path = "C:/D/PatientData/Trajectories"
    add_couch_angle_info(folder_path)

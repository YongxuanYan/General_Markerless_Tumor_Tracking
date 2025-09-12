import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import glob
import shutil

#使用光流法追踪X透视图像的基准标记轨迹，会生成已处理的列表以跳过，以及轨迹CSV，选定的基准标记的绿色方框引导

# ===== 配置参数 =====
BASE_DIR = r"C:\D\PatientData\CFI"  # 基础路径
OUTPUT_DIR = r"C:\D\PatientData\Trajectories"  # 输出目录
MARKERS_DIR = os.path.join(OUTPUT_DIR, "Markers")  # 标记图像输出目录
PROCESSED_LOG = os.path.join(OUTPUT_DIR, "processed_sequences.txt")  # 处理记录文件


def load_processed_sequences():
    """加载已处理的序列记录"""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_sequence_processed(sequence_id):
    """标记序列为已处理"""
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{sequence_id}\n")


def get_user_selected_point(r_channel):
    """显示R通道并让用户手动选择标记点"""
    plt.ion()  # 启用交互模式
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(r_channel, cmap='gray')
    ax.set_title("Please select marker (black dot)\n and then close the window")
    ax.axis('on')

    print("\nPlease select marker dot")
    selected_point = plt.ginput(1, timeout=0)[0]
    plt.close()
    return selected_point  # 返回(x,y)元组


def save_marked_image(img_path, point, output_path):
    """保存带标记点的图像"""
    # 读取原始图像
    img = cv2.imread(img_path)
    if img is None:
        return False

    # 转换为RGB并绘制标记
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y = int(point[0]), int(point[1])

    # 绘制10x10绿色方框
    cv2.rectangle(img_rgb, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)

    # 保存图像
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return True


def track_single_sequence(img_files, sequence_id):
    """处理单个序列并保存标记图像"""
    # 创建标记图像输出目录
    marker_output_dir = os.path.join(MARKERS_DIR, sequence_id.replace('/', '\\'))
    os.makedirs(marker_output_dir, exist_ok=True)

    # 读取第一帧（提取R通道）
    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"can not read the first frame {img_files[0]}")
        return None

    r_channel = first_frame[:, :, 0]  # OpenCV是BGR顺序，索引2是R通道

    # 用户手动选择标记点
    try:
        initial_point = get_user_selected_point(r_channel)
        print(f"Selected point ({initial_point[0]:.1f}, {initial_point[1]:.1f})")
    except Exception as e:
        print(f"Failed {e}")
        return None

    # 保存第一帧标记图像
    first_output = os.path.join(marker_output_dir, os.path.basename(img_files[0]))
    save_marked_image(img_files[0], initial_point, first_output)

    # 准备光流追踪（使用反相的R通道）
    inverted_r = 255 - r_channel  # 反相处理，使黑点变白
    trajectory = [initial_point]
    prev_pt = np.array([initial_point], dtype=np.float32).reshape(-1, 1, 2)
    prev_img = inverted_r

    lk_params = dict(
        winSize=(8, 8),  # 增大窗口尺寸以捕获更多上下文信息
        maxLevel=2,  # 增加金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
    )

    # 处理后续帧 - 修改了tqdm参数
    for i, path in enumerate(tqdm(img_files[1:], desc="Progress", leave=False)):
        curr_frame = cv2.imread(path)
        if curr_frame is None:
            print(f"Warning: can not read {os.path.basename(path)}")
            trajectory.append(trajectory[-1])  # 使用上一位置
            # 保存错误帧标记
            if trajectory:
                save_marked_image(path, trajectory[-1],
                                  os.path.join(marker_output_dir, os.path.basename(path)))
            continue

        curr_img = 255 - curr_frame[:, :, 0]  # 反相R通道
        curr_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img, curr_img, prev_pt, None, **lk_params)

        # 更新轨迹
        if status[0] == 1:
            x, y = curr_pt.ravel()  # 直接解包坐标
            trajectory.append((x, y))
        else:
            trajectory.append(trajectory[-1])

        # 保存当前帧标记图像
        save_marked_image(path, trajectory[-1],
                          os.path.join(marker_output_dir, os.path.basename(path)))

        # 更新参考
        prev_img = curr_img
        prev_pt = curr_pt if status[0] == 1 else prev_pt

    # 保存本序列结果
    result_df = pd.DataFrame([
        {'frame': idx, 'marker_x': x, 'marker_y': y}
        for idx, (x, y) in enumerate(trajectory)
    ])
    return result_df


def process_all_sequences():
    """处理所有患者数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MARKERS_DIR, exist_ok=True)
    processed = load_processed_sequences()
    patient_dirs = natsorted(glob.glob(os.path.join(BASE_DIR, "PT*")))

    for p_dir in tqdm(patient_dirs, desc="Progress"):
        patient_id = os.path.basename(p_dir)
        patient_results = []

        # 遍历F文件夹
        for f_dir in natsorted(glob.glob(os.path.join(p_dir, "F*"))):
            field_id = os.path.basename(f_dir)

            # 处理两个grab文件夹
            for grab_dir in [os.path.join(f_dir, "grab0"), os.path.join(f_dir, "grab1")]:
                if not os.path.exists(grab_dir):
                    continue

                # 构建唯一序列ID
                device_id = os.path.basename(grab_dir)
                sequence_id = f"{patient_id}/{field_id}/{device_id}"

                # 跳过已处理的序列
                if sequence_id in processed:
                    print(f"Skip processed sequence {sequence_id}")
                    continue

                # 获取图像文件
                img_files = natsorted(
                    glob.glob(os.path.join(grab_dir, "*.png")) +
                    glob.glob(os.path.join(grab_dir, "*.tif")) +
                    glob.glob(os.path.join(grab_dir, "*.jpg")),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )

                if not img_files:
                    print(f"Sequence {sequence_id} Has no image")
                    continue

                print(f"\nProcessing sequence {sequence_id}")
                print(f"Image number{len(img_files)}")

                # 处理当前序列
                seq_df = track_single_sequence(img_files, sequence_id)
                if seq_df is None:
                    print(">> Process failed, skipping")
                    continue

                # 添加元数据并保存
                seq_df['patient'] = patient_id
                seq_df['field'] = field_id
                seq_df['device'] = device_id

                output_file = os.path.join(OUTPUT_DIR, f"{sequence_id.replace('/', '_')}.csv")
                seq_df.to_csv(output_file, index=False)
                mark_sequence_processed(sequence_id)
                print(f">> saved to: {output_file}")

    print("\nAll sequences processed!")


if __name__ == "__main__":
    process_all_sequences()
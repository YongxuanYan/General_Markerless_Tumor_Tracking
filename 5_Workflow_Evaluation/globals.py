import os
import json
import tempfile
import atexit
import numpy as np

# 创建一个临时文件夹，用于存储全局变量的 JSON 文件
TEMP_DIR = 'C:/Users/25165/PycharmProjects/Evaluate_2nd_Paper'  # 自动管理临时文件夹
GLOBAL_VARS_FILE = os.path.join(TEMP_DIR, "global_vars.json")

# 初始化全局变量
GLOBAL_VARS = {
    "ColorInputs": False,
    "CouchAngle": None,
    "CurrentSlice": 1,
    "CurrentDRRSlice": 1,
    "CurrentAiSlice": 1,
    "ctexist": 0,
    "CT_MAX_HU": 0,
    "DRREXISTS_Left": 0,
    "DRREXISTS_Right": 0,
    "DRR_Resolution": None,
    "DRRs": None,
    "Geoinfo_save_path": None,
    "Ground_truth_path": None,
    "I0": 1200,
    "ISO_X": None,
    "ISO_Y": None,
    "ISO_Z": None,
    "ImagingPair": None,
    "Imaging_pair_1_enabled": 0,
    "Imaging_pair_2_enabled": 0,
    "Imaging_pair_1_fileName": None,
    "Imaging_pair_2_fileName": None,
    "labeldata": None,
    "labelexits": 0,
    "Left_DRR": None,
    "LocationOfNotCTData": None,
    "MaxWindowLevel": 3071,
    "MaxWindowWidth": 4095,
    "MinWindowLevel": -1024,
    "Model_architecture_path": None,
    "Model_evaluation_path": None,
    "Model_inputs_filenames": None,
    "Model_inputs_images": None,
    "Model_inputs_images_exists": False,
    "Model_inputs_path": None,
    "Model_loading_msg": '',
    "Model_outputs_images": None,
    "Model_outputs_images_exists": False,
    "Model_outputs_path": None,
    "Model_weights_path": None,
    "NoMoreAsking": False,
    "PDepth": None,
    "PHeight": None,
    "PWidth": None,
    "PatientName": None,
    "PixelSpacing": None,
    "PixelsGrid": None,
    "Right_DRR": None,
    "SliceLocation": None,
    "SliceNum": 0,
    "SliceThickness": None,
    "SysMSG": "Welcome to YU LAB-B504.",
    "TileSize": None,
    "TotalModelInputsNum": 0,
    "WindowLevel": -500,
    "WindowWidth": 1500,
}


# 初始化 JSON 文件
def initialize_global_vars():
    """初始化 JSON 文件，将全局变量写入文件"""
    with open(GLOBAL_VARS_FILE, 'w') as f:
        json.dump(GLOBAL_VARS, f, indent=4)


# 在模块导入时自动初始化
initialize_global_vars()


# 设置全局变量
def set_var(var_name, value):
    """设置全局变量的值"""
    # 首先读取现有的全局变量
    with open(GLOBAL_VARS_FILE, 'r') as f:
        global_vars = json.load(f)

    # 对于 ndarray 类型的数据，单独保存为 .npy 文件
    if isinstance(value, np.ndarray):
        npy_path = os.path.join(TEMP_DIR, f"{var_name}.npy")
        np.save(npy_path, value)
        value = {"type": "ndarray", "path": npy_path}

    # 更新变量值
    global_vars[var_name] = value

    # 写回文件
    with open(GLOBAL_VARS_FILE, 'w') as f:
        json.dump(global_vars, f, indent=4)


def get_var(var_name, default=None):
    """获取全局变量的值，如果不存在则返回默认值"""
    try:
        with open(GLOBAL_VARS_FILE, 'r') as f:
            global_vars = json.load(f)

        if var_name in global_vars:
            # 如果是 ndarray 的路径，加载 .npy 文件
            npy_path = os.path.join(TEMP_DIR, "PixelsGrid.npy")
            if os.path.exists(npy_path):
                value = np.load(npy_path)
            else:
                raise FileNotFoundError(f"File {npy_path} does not exist.")
            return value
        else:
            return default
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def del_var(var_name):
    """删除全局变量"""
    # 首先读取现有的全局变量
    with open(GLOBAL_VARS_FILE, 'r') as f:
        global_vars = json.load(f)

    # 删除变量
    if var_name in global_vars:
        del global_vars[var_name]

        # 写回文件
        with open(GLOBAL_VARS_FILE, 'w') as f:
            json.dump(global_vars, f, indent=4)

    # 删除对应的 .npy 文件（如果存在）
    npy_path = os.path.join(TEMP_DIR, f"{var_name}.npy")
    if os.path.exists(npy_path):
        os.remove(npy_path)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import os
import json
from globals import get_var, set_var  # 你拷贝的globals.py

# 固定临时目录路径
TEMP_DIR = 'C:/Users/25165/PycharmProjects/Evaluate_2nd_Paper'
os.makedirs(TEMP_DIR, exist_ok=True)


class CTEditor:
    def __init__(self):
        self.ct_data = None
        self.current_slice = 164  # 默认显示第100层
        self.clicked_points = []  # 存储点击的点
        self.modification_radius = 5  # 默认修改半径5mm
        self.voxel_value = 3000  # 默认修改值
        self.fig = None
        self.ax = None
        self.img_display = None

        self.load_ct_data()
        self.create_interface()

    def load_ct_data(self):
        """从全局变量加载CT数据"""
        try:
            self.ct_data = get_var("PixelsGrid")
            if self.ct_data is None:
                print("错误：未找到CT数据")
                return False

            print(f"CT数据加载成功，形状: {self.ct_data.shape}")
            return True

        except Exception as e:
            print(f"加载CT数据失败: {str(e)}")
            return False

    def create_interface(self):
        """创建交互界面"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)

        # 显示当前切片
        self.update_display()

        # 添加切片选择滑块
        ax_slice = plt.axes([0.1, 0.15, 0.8, 0.03])
        self.slice_slider = Slider(
            ax_slice, 'Slice', 1, self.ct_data.shape[2],
            valinit=self.current_slice, valstep=1
        )
        self.slice_slider.on_changed(self.on_slice_changed)

        # 添加半径选择滑块
        ax_radius = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.radius_slider = Slider(
            ax_radius, 'radius', 1, 20,
            valinit=self.modification_radius, valstep=0.5
        )
        self.radius_slider.on_changed(self.on_radius_changed)

        # 添加体素值输入框
        ax_value = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.value_slider = Slider(
            ax_value, 'Voxiel value', -1000, 10000,
            valinit=self.voxel_value, valstep=100
        )
        self.value_slider.on_changed(self.on_value_changed)

        # 添加按钮
        ax_apply = plt.axes([0.1, 0.01, 0.2, 0.03])
        self.apply_button = Button(ax_apply, 'apply')
        self.apply_button.on_clicked(self.on_apply_clicked)

        ax_save = plt.axes([0.4, 0.01, 0.2, 0.03])
        self.save_button = Button(ax_save, 'save')
        self.save_button.on_clicked(self.on_save_clicked)

        ax_clear = plt.axes([0.7, 0.01, 0.2, 0.03])
        self.clear_button = Button(ax_clear, 'clear')
        self.clear_button.on_clicked(self.on_clear_clicked)

        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_image_clicked)

        plt.title(f"CT editor - slice {self.current_slice}/{self.ct_data.shape[2]}")
        plt.show()

    def update_display(self):
        """更新CT图像显示"""
        if self.ct_data is None:
            return

        # 获取当前切片（转换为0-based索引）
        slice_idx = self.current_slice - 1
        slice_data = self.ct_data[:, :, slice_idx]

        # 清除之前的显示
        if self.img_display:
            self.img_display.remove()

        # 使用合适的窗宽窗位显示
        window_level = -500
        window_width = 1500
        vmin = window_level - window_width // 2
        vmax = window_level + window_width // 2

        # 显示图像
        self.img_display = self.ax.imshow(
            slice_data,
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            origin='lower'
        )

        # 绘制之前点击的点
        for point in self.clicked_points:
            if point[2] == slice_idx:  # 只显示当前切片的点
                circle = Circle((point[1], point[0]), 10, fill=False, color='red', linewidth=2)
                self.ax.add_patch(circle)
                self.ax.plot(point[1], point[0], 'ro', markersize=4)

        self.ax.set_title(f"Slice {self.current_slice}/{self.ct_data.shape[2]} - Click to select point")
        self.fig.canvas.draw()

    def on_slice_changed(self, val):
        """切片变化时的处理"""
        self.current_slice = int(val)
        self.update_display()

    def on_radius_changed(self, val):
        """修改半径变化时的处理"""
        self.modification_radius = val

    def on_value_changed(self, val):
        """体素值变化时的处理"""
        self.voxel_value = int(val)

    def on_image_clicked(self, event):
        """处理图像点击事件"""
        if event.inaxes != self.ax:
            return

        # 获取点击位置（转换为数组坐标）
        x = int(event.ydata)
        y = int(event.xdata)
        slice_idx = self.current_slice - 1

        # 添加到点击点列表
        self.clicked_points.append((x, y, slice_idx))
        print(f"Selected Slice {self.current_slice}, Position({x}, {y})")

        # 立即显示修改效果（绘制圆圈标记）
        circle = Circle((y, x), 10, fill=False, color='red', linewidth=2)
        self.ax.add_patch(circle)
        self.ax.plot(y, x, 'ro', markersize=4)
        self.fig.canvas.draw()

    def on_apply_clicked(self, event):
        """应用修改到CT数据"""
        if not self.clicked_points:
            print("Please select point first")
            return

        voxel_radius = int(self.modification_radius)  # 1mm层厚，半径对应体素数

        for point in self.clicked_points:
            x, y, z = point
            self.modify_voxels(x, y, z, voxel_radius)

        print(f"Applied to {len(self.clicked_points)} voxiels")
        self.update_display()

    def modify_voxels(self, x, y, z, radius):
        """修改指定位置的体素值"""
        # 在3D空间中创建球形掩模
        z_center = z
        y_center = y
        x_center = x

        # 创建坐标网格
        z_range = range(max(0, z_center - radius), min(self.ct_data.shape[2], z_center + radius + 1))
        y_range = range(max(0, y_center - radius), min(self.ct_data.shape[1], y_center + radius + 1))
        x_range = range(max(0, x_center - radius), min(self.ct_data.shape[0], x_center + radius + 1))

        modified_count = 0
        # 应用球形修改
        for z_idx in z_range:
            for y_idx in y_range:
                for x_idx in x_range:
                    # 计算距离（欧几里得距离）
                    distance = np.sqrt((x_idx - x_center) ** 2 + (y_idx - y_center) ** 2 + (z_idx - z_center) ** 2)
                    if distance <= radius:
                        self.ct_data[x_idx, y_idx, z_idx] = self.voxel_value
                        modified_count += 1

        print(f"Edit point ({x},{y},{z}) area {modified_count} voxiels to {self.voxel_value}")

    def on_save_clicked(self, event):
        """保存修改后的CT数据"""
        try:
            set_var("PixelsGrid", self.ct_data)
            print("CT data saved！")
            print(f"path: {TEMP_DIR}")
        except Exception as e:
            print(f"failed {str(e)}")

    def on_clear_clicked(self, event):
        """清除所有选择点"""
        self.clicked_points = []
        print("cleared all points")
        self.update_display()


# 运行编辑器
if __name__ == "__main__":
    print("boosting")
    print(f"path:  {TEMP_DIR}")

    editor = CTEditor()
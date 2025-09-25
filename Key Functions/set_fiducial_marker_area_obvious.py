"""
This script allows interactive modification of CT data to enhance fiducial markers.
Requires CT data to be converted to numpy array and saved as .npy file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import os
import json
from globals import get_var, set_var

# Temporary directory path
TEMP_DIR = 'path/to/CT_data.npy/directory'
os.makedirs(TEMP_DIR, exist_ok=True)


class CTEditor:
    """Interactive CT data editor for enhancing fiducial markers."""
    
    def __init__(self):
        self.ct_data = None
        self.current_slice = 164  # Default slice
        self.clicked_points = []  # Store clicked points
        self.modification_radius = 5  # Default radius in mm
        self.voxel_value = 3000  # Default modification value
        self.fig = None
        self.ax = None
        self.img_display = None

        self.load_ct_data()
        self.create_interface()

    def load_ct_data(self):
        """Load CT data from global variables."""
        try:
            self.ct_data = get_var("PixelsGrid")
            if self.ct_data is None:
                print("Error: CT data not found")
                return False

            print(f"CT data loaded successfully, shape: {self.ct_data.shape}")
            return True

        except Exception as e:
            print(f"Failed to load CT data: {str(e)}")
            return False

    def create_interface(self):
        """Create the interactive interface."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)

        self.update_display()

        # Slice selection slider
        ax_slice = plt.axes([0.1, 0.15, 0.8, 0.03])
        self.slice_slider = Slider(
            ax_slice, 'Slice', 1, self.ct_data.shape[2],
            valinit=self.current_slice, valstep=1
        )
        self.slice_slider.on_changed(self.on_slice_changed)

        # Radius selection slider
        ax_radius = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.radius_slider = Slider(
            ax_radius, 'Radius (mm)', 1, 20,
            valinit=self.modification_radius, valstep=0.5
        )
        self.radius_slider.on_changed(self.on_radius_changed)

        # Voxel value slider
        ax_value = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.value_slider = Slider(
            ax_value, 'Voxel Value', -1000, 10000,
            valinit=self.voxel_value, valstep=100
        )
        self.value_slider.on_changed(self.on_value_changed)

        # Control buttons
        ax_apply = plt.axes([0.1, 0.01, 0.2, 0.03])
        self.apply_button = Button(ax_apply, 'Apply')
        self.apply_button.on_clicked(self.on_apply_clicked)

        ax_save = plt.axes([0.4, 0.01, 0.2, 0.03])
        self.save_button = Button(ax_save, 'Save')
        self.save_button.on_clicked(self.on_save_clicked)

        ax_clear = plt.axes([0.7, 0.01, 0.2, 0.03])
        self.clear_button = Button(ax_clear, 'Clear')
        self.clear_button.on_clicked(self.on_clear_clicked)

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_image_clicked)

        plt.title(f"CT Editor - Slice {self.current_slice}/{self.ct_data.shape[2]}")
        plt.show()

    def update_display(self):
        """Update the CT image display."""
        if self.ct_data is None:
            return

        slice_idx = self.current_slice - 1
        slice_data = self.ct_data[:, :, slice_idx]

        if self.img_display:
            self.img_display.remove()

        # Display with appropriate window settings
        window_level = -500
        window_width = 1500
        vmin = window_level - window_width // 2
        vmax = window_level + window_width // 2

        self.img_display = self.ax.imshow(
            slice_data,
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            origin='lower'
        )

        # Draw previously clicked points
        for point in self.clicked_points:
            if point[2] == slice_idx:
                circle = Circle((point[1], point[0]), 10, fill=False, color='red', linewidth=2)
                self.ax.add_patch(circle)
                self.ax.plot(point[1], point[0], 'ro', markersize=4)

        self.ax.set_title(f"Slice {self.current_slice}/{self.ct_data.shape[2]} - Click to select point")
        self.fig.canvas.draw()

    def on_slice_changed(self, val):
        """Handle slice change."""
        self.current_slice = int(val)
        self.update_display()

    def on_radius_changed(self, val):
        """Handle radius change."""
        self.modification_radius = val

    def on_value_changed(self, val):
        """Handle voxel value change."""
        self.voxel_value = int(val)

    def on_image_clicked(self, event):
        """Handle image click events."""
        if event.inaxes != self.ax:
            return

        x = int(event.ydata)
        y = int(event.xdata)
        slice_idx = self.current_slice - 1

        self.clicked_points.append((x, y, slice_idx))
        print(f"Selected Slice {self.current_slice}, Position({x}, {y})")

        # Draw marker immediately
        circle = Circle((y, x), 10, fill=False, color='red', linewidth=2)
        self.ax.add_patch(circle)
        self.ax.plot(y, x, 'ro', markersize=4)
        self.fig.canvas.draw()

    def on_apply_clicked(self, event):
        """Apply modifications to CT data."""
        if not self.clicked_points:
            print("Please select points first")
            return

        voxel_radius = int(self.modification_radius)

        for point in self.clicked_points:
            x, y, z = point
            self.modify_voxels(x, y, z, voxel_radius)

        print(f"Applied modifications to {len(self.clicked_points)} regions")
        self.update_display()

    def modify_voxels(self, x, y, z, radius):
        """Modify voxel values in spherical region."""
        z_center = z
        y_center = y
        x_center = x

        # Create coordinate grid
        z_range = range(max(0, z_center - radius), min(self.ct_data.shape[2], z_center + radius + 1))
        y_range = range(max(0, y_center - radius), min(self.ct_data.shape[1], y_center + radius + 1))
        x_range = range(max(0, x_center - radius), min(self.ct_data.shape[0], x_center + radius + 1))

        modified_count = 0
        # Apply spherical modification
        for z_idx in z_range:
            for y_idx in y_range:
                for x_idx in x_range:
                    distance = np.sqrt((x_idx - x_center) ** 2 + (y_idx - y_center) ** 2 + (z_idx - z_center) ** 2)
                    if distance <= radius:
                        self.ct_data[x_idx, y_idx, z_idx] = self.voxel_value
                        modified_count += 1

        print(f"Modified {modified_count} voxels at ({x},{y},{z}) to value {self.voxel_value}")

    def on_save_clicked(self, event):
        """Save modified CT data."""
        try:
            set_var("PixelsGrid", self.ct_data)
            print("CT data saved successfully")
            print(f"Path: {TEMP_DIR}")
        except Exception as e:
            print(f"Save failed: {str(e)}")

    def on_clear_clicked(self, event):
        """Clear all selected points."""
        self.clicked_points = []
        print("Cleared all points")
        self.update_display()


if __name__ == "__main__":
    print("Starting CT Editor...")
    print(f"Temporary directory: {TEMP_DIR}")
    editor = CTEditor()

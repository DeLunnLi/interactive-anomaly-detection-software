import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal,QObject  # 导入 QtCore 用于设置鼠标形状
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class InteractivePlot(QObject):
    two_threshold = pyqtSignal(float,float)
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.labeled_normal_scores = np.array([])
        self.labeled_abnormal_scores = np.array([])
        self.bins = np.linspace(0, 1, 300)
        self.width = 0
        self.classification = False
        self.init_plot()

    def init_plot(self):
        """Initialize the plot without initial plot data and setup mouse events"""
        # Set x and y limits to 0 to 1 range for an empty initial plot
        self.canvas.ax.set_xlim([0, 1])
        self.canvas.ax.set_ylim([0, 1])

        # Hide y-axis labels and ticks, and remove x-axis ticks except for min and max
        self.canvas.ax.set_yticklabels([])
        self.canvas.ax.tick_params(axis='y', which='both', length=0)
        self.canvas.ax.tick_params(axis='x', which='both', length=0)
        self.canvas.ax.set_xticks([0, 1])
        self.width = 1
        # Set y-axis label for score
        self.canvas.ax.set_ylabel("Count")  # Add label for the y-axis

        # Set x-axis label for count
        self.canvas.ax.set_xlabel("Score")  # Add label for the x-axis

        # Initial positions of the draggable lines with dashed linestyle
        self.line1 = self.canvas.ax.axvline(x=0.75, color='r', lw=2, linestyle='-.')
        self.line2 = self.canvas.ax.axvline(x=0.25, color='g', lw=2, linestyle='-.')
        
        # Text labels to display x-coordinates
        self.label1 = self.canvas.ax.text(0.75, 0.5, '', color='red', fontsize=12, ha='center')
        self.label2 = self.canvas.ax.text(0.25, 0.5, '', color='green', fontsize=12, ha='center')

        # Fill between the two lines (initial shading)
        self.fill_between_lines = self.canvas.ax.fill_betweenx(
            y=[0, 1], x1=0.25, x2=0.75, color='gray', alpha=0.5
        )

        # Connect the canvas with mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.selected_line = None

        # Initial update of labels
        self.update_labels()

    def update_plot(self,labeled_normal_scores,labeled_abnormal_scores):
        if labeled_normal_scores.size == 0 and labeled_abnormal_scores.size == 0:
            return
        self.classification = True
        self.canvas.ax.clear()

        normal_score_min = None if labeled_normal_scores.size == 0 else labeled_normal_scores.min()
        normal_score_max = None if labeled_normal_scores.size == 0 else labeled_normal_scores.max()
        abnormal_score_min = None if labeled_abnormal_scores.size == 0 else labeled_abnormal_scores.min()
        abnormal_score_max = None if labeled_abnormal_scores.size == 0 else labeled_abnormal_scores.max()
        min_score = min(score for score in [normal_score_min,abnormal_score_min] if score is not None)
        max_score = max(score for score in [normal_score_max,abnormal_score_max] if score is not None)
        line1_x = None
        line2_x = None
        self.canvas.ax.set_xlim([0.95 * min_score,min(1.1 * max_score,1)])
        self.width = max_score - min_score
        self.canvas.ax.set_ylim([0, 1])
        self.canvas.ax.set_yticklabels([])
        self.canvas.ax.tick_params(axis='y', which='both', length=0)
        self.canvas.ax.tick_params(axis='x', which='both', length=0)
        self.canvas.ax.set_xticks([round(0.95 * min_score,3), min(round(1.1 * max_score,3),1)])
        # self.bins = np.linspace(round(0.95* min_score,3),  round(1.05*max_score,3), 300)
        # Set y-axis label for score
        self.canvas.ax.set_ylabel("Count")  # Add label for the y-axis

        # Set x-axis label for count
        self.canvas.ax.set_xlabel("Score")  # Add label for the x-axis

        if labeled_normal_scores.size != 0:
            line2_x = normal_score_max
            normal_x,normal_y = self.complementary_cumulative_percentage(labeled_normal_scores)
            normal_x = np.insert(normal_x, 0, round(0.95*min(normal_x),3))
            normal_y = np.insert(normal_y, 0, 1)
            self.canvas.ax.plot(normal_x, normal_y, color='#32CD32')
            self.canvas.ax.fill_between(normal_x, normal_y, color='#00FF00', alpha=0.3)
        if labeled_abnormal_scores.size != 0:
            line1_x = abnormal_score_min
            abnormal_x,abnormal_y = self.cumulative_percentage(labeled_abnormal_scores)
            abnormal_x = np.append(abnormal_x,min(round(1.1*max(abnormal_x),3),1))
            abnormal_y = np.append(abnormal_y,1)
            self.canvas.ax.plot(abnormal_x, abnormal_y, color='red')
            self.canvas.ax.fill_between(abnormal_x, abnormal_y, color='red', alpha=0.3)
        
        if line1_x==None:
            line1_x = line2_x
        elif line2_x == None:
            line2_x = line1_x

        # Initial positions of the draggable lines with dashed linestyle
        self.line1 = self.canvas.ax.axvline(x=line1_x, color='r', lw=2, linestyle='-.')
        self.line2 = self.canvas.ax.axvline(x=line2_x, color='g', lw=2, linestyle='-.')
        
        # Text labels to display x-coordinates
        self.label1 = self.canvas.ax.text(line1_x, 0.5, '', color='red', fontsize=12, ha='center')
        self.label2 = self.canvas.ax.text(line2_x, 0.5, '', color='green', fontsize=12, ha='center')

        # Fill between the two lines (initial shading)
        self.fill_between_lines = self.canvas.ax.fill_betweenx(
            y=[0, 1], x1=line2_x, x2=line1_x, color='gray', alpha=0.5
        )

        # Connect the canvas with mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.selected_line = None

        red_x = self.line1.get_xdata()[0]
        green_x = self.line2.get_xdata()[0]

        # 检查红线是否在绿线的左侧，如果是则交换它们的位置
        if red_x < green_x:
            # 交换位置
            self.line1.set_xdata(green_x)
            self.line2.set_xdata(red_x)

        # Initial update of labels
        self.update_labels()
        self.update_fill_between_lines()  # 更新两线之间的填充
        self.canvas.draw()
        self.two_threshold.emit(green_x,red_x)

    # 使用bin的方式进行统计
    # def cumulative_percentage(self, data):
    #     counts, bin_edges = np.histogram(data, bins=self.bins)
    #     cdf = np.cumsum(counts) / counts.sum()
    #     return bin_edges[1:], cdf

    # def complementary_cumulative_percentage(self, data):
    #     counts, bin_edges = np.histogram(data, bins=self.bins)
    #     cdf = np.cumsum(counts) / counts.sum()
    #     return bin_edges[1:], 1 - cdf

    def cumulative_percentage(self, data):
        """Calculates cumulative percentage with range from 0 to 1."""
        sorted_data = np.sort(data)
        if len(sorted_data) == 1:
            cumulative =np.zeros_like(sorted_data)
        else:
            cumulative = np.linspace(0, 1, len(sorted_data))  # 使用 linspace 使其从 0 到 1
        return sorted_data, cumulative

    def complementary_cumulative_percentage(self, data):
        """Calculates complementary cumulative percentage with range from 1 to 0."""
        sorted_data = np.sort(data)
        complementary_cumulative = np.linspace(1, 0, len(sorted_data))  # 使用 linspace 使其从 1 到 0
        return sorted_data, complementary_cumulative

    def on_click(self, event):
        """Check if a line is selected on mouse click"""
        if event.xdata is None:
            return

        # Determine if click is near a line (line 1 or line 2)
        if abs(event.xdata - self.line1.get_xdata()[0]) < 0.02 * self.width:
            self.selected_line = self.line1
        elif abs(event.xdata - self.line2.get_xdata()[0]) < 0.02 * self.width:
            self.selected_line = self.line2
        else:
            self.selected_line = None

    def on_mouse_move(self, event):
        """Move the selected line with mouse movement"""
        if event.xdata is None:
            return

        # Change cursor when mouse is near the lines
        if abs(event.xdata - self.line1.get_xdata()[0]) < 0.02 * self.width or abs(event.xdata - self.line2.get_xdata()[0]) < 0.02 * self.width:
            self.canvas.setCursor(Qt.SplitHCursor)  # Set to left-right arrow cursor
        else:
            self.canvas.setCursor(Qt.ArrowCursor)  # Reset to default cursor

        if self.selected_line is not None:
            # Move the selected line to the mouse position
            self.selected_line.set_xdata(event.xdata)
            self.update_labels()  # Update labels whenever the line is moved
            self.update_fill_between_lines()  # Update the shading between the lines
            self.canvas.draw()

    def on_release(self, event):
        red_x = self.line1.get_xdata()[0]
        green_x = self.line2.get_xdata()[0]

        # 检查红线是否在绿线的左侧，如果是则交换它们的位置
        if red_x < green_x:
            # 交换位置
            self.line1.set_xdata(green_x)
            self.line2.set_xdata(red_x)
        
        self.selected_line = None  # 释放选中的线
        self.update_labels()  # 更新标签
        self.update_fill_between_lines()  # 更新两线之间的填充
        self.canvas.draw()  # 重新绘制画布
        if self.classification:
            self.two_threshold.emit(self.line2.get_xdata()[0],self.line1.get_xdata()[0])

    def update_labels(self):
        """Update the position and text of the labels"""
        x1 = self.line1.get_xdata()[0]
        x2 = self.line2.get_xdata()[0]

        self.label1.set_position((x1, 1.0))  # Move label slightly down for the red line
        self.label1.set_ha('left')           # Right-align the label
        self.label1.set_text(f'{x1:.3f}')     # Set text to display x-coordinate

        self.label2.set_position((x2, 1.0))  # Move label slightly down for the green line
        self.label2.set_ha('right')           # Right-align the label
        self.label2.set_text(f'{x2:.3f}')     # Set text to display x-coordinate

    def update_fill_between_lines(self):
        """Update the shaded area between the two lines"""
        if self.fill_between_lines:
            self.fill_between_lines.remove()

        x1 = self.line1.get_xdata()[0]
        x2 = self.line2.get_xdata()[0]

        self.fill_between_lines = self.canvas.ax.fill_betweenx(
            y=[0, 1], x1=x1, x2=x2, color='gray', alpha=0.5
        )
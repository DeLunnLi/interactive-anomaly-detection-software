import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from .interactive_plot import MplCanvas,InteractivePlot
from .confusion_matrix import ConfusionMatrix

class Interactive_statistics(QWidget):  # 从 QWidget 继承
    def __init__(self):
        super().__init__()

        # 创建主布局
        self.main_layout = QVBoxLayout(self)

        # 数量行布局（与第一行相对应）
        count_layout = QHBoxLayout()

        # 创建与第一行相对应的数量标签并添加到布局中，只显示数字
        self.image_count_label = QLabel('0')
        self.labeled_count_label = QLabel('0')
        self.trained_count_label = QLabel('0')

        # 设置对齐方式，保证数量标签和下面的汉字居中对齐
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.labeled_count_label.setAlignment(Qt.AlignCenter)
        self.trained_count_label.setAlignment(Qt.AlignCenter)

        # 增大数量标签的字号
        self.image_count_label.setStyleSheet("font-size: 18px;font-weight: bold;")
        self.labeled_count_label.setStyleSheet("font-size: 18px;font-weight: bold;")
        self.trained_count_label.setStyleSheet("font-size: 18px;font-weight: bold;")

        # 将三个数字标签添加到布局中
        count_layout.addWidget(self.image_count_label)
        count_layout.addWidget(self.labeled_count_label)
        count_layout.addWidget(self.trained_count_label)

        # 第一行布局（图像、已标记、已训练）
        top_layout = QHBoxLayout()

        # 创建三个标签并添加到布局中
        image_label = QLabel('Image')
        labeled_label = QLabel('Labeled')
        trained_label = QLabel('Trained')

        # 设置汉字标签对齐方式，使其和数量标签保持垂直居中对齐
        image_label.setAlignment(Qt.AlignCenter)
        labeled_label.setAlignment(Qt.AlignCenter)
        trained_label.setAlignment(Qt.AlignCenter)

        # 增大汉字标签的字号
        image_label.setStyleSheet("font-size: 16px;font-weight: bold;")
        labeled_label.setStyleSheet("font-size: 16px;font-weight: bold;")
        trained_label.setStyleSheet("font-size: 16px;font-weight: bold;")

        # 将三个汉字标签添加到布局中
        top_layout.addWidget(image_label)
        top_layout.addWidget(labeled_label)
        top_layout.addWidget(trained_label)

        # 第二行布局（处理时间居中）
        time_layout = QHBoxLayout()
        # 不再使用动态高度填充，而是使用固定的空白高度
        self.processing_time_label = QLabel('Average Processing Time：00 ms')
        time_layout.addStretch()  # 左侧填充空间
        time_layout.addWidget(self.processing_time_label)  # 添加标签
        time_layout.addStretch()  # 右侧填充空间

        # 增大处理时间标签的字号
        self.processing_time_label.setStyleSheet("font-size: 16px;font-weight: bold;")

        self.plot_label = QLabel('Cumulative Plot')
        self.plot_label.setAlignment(Qt.AlignLeft)  # 右对齐
        self.plot_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        # 新建一个水平布局用于放置标签和图像
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot_label)  # 添加标签
        plot_layout.addStretch()  # 左侧填充空间

        # 添加到布局中
        self.main_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # 固定的间距
        self.main_layout.addLayout(count_layout)  # 添加数量行到主布局
        self.main_layout.addLayout(top_layout)    # 添加第一行
        self.main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))  # 固定的间距
        self.main_layout.addLayout(time_layout)   # 添加处理时间行

        # 在处理时间和图像之间添加固定的间距
        self.main_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))  # 固定的间距
        self.main_layout.addLayout(plot_layout)
        # 第三行布局（显示图像)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        self.main_layout.addWidget(self.canvas)    # 添加图像显示区域
        self.plot = InteractivePlot(self.canvas)

        # 在图像和混淆矩阵之间添加固定的间距
        self.main_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))  # 固定的间距

        # 第四行布局（混淆矩阵）
        confusion_matrix_label = QLabel('Confusion Matrix')
        confusion_matrix_label.setAlignment(Qt.AlignLeft)
        confusion_matrix_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(confusion_matrix_label)

        self.confusion_matrix = ConfusionMatrix()
        self.main_layout.addWidget(self.confusion_matrix)       # 添加混淆矩阵

        # 添加底部的空白空间，确保整体布局的均匀性
        self.main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))  # 底部扩展的间距

    def update_image_count(self,image_count):
        self.image_count_label.setText(str(image_count))

    def update_labeled_count(self,labeled_count):
        self.labeled_count_label.setText(str(labeled_count))

    def update_trained_count(self,trained_count):
        self.trained_count_label.setText(str(trained_count))

    def update_processing_time(self, average):
        self.processing_time_label.setText(f'Average Processing Time：{average:.2f} ms')
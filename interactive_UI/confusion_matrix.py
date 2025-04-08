import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt

class ConfusionMatrix(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("混淆矩阵")

        # 主布局
        main_layout = QVBoxLayout()

        # 预期标签布局
        expectation_layout = QHBoxLayout()
        expectation_layout.setSpacing(0)
        expectation_layout.addStretch(9)  # 左侧空间
        expectation_label = QLabel("Predict")
        expectation_label.setStyleSheet("font-size: 20px;font-weight: bold;")
        expectation_layout.addWidget(expectation_label)
        expectation_layout.addStretch(4)  # 右侧空间
        main_layout.addLayout(expectation_layout)

        line_container = QWidget()
        line_layout = QVBoxLayout(line_container)  # 将line_layout设置为line_container的布局

        # 创建一个水平布局用于放置空格和横线
        h_layout = QHBoxLayout()

        # 添加一个空的QWidget作为左边的空格
        spacer = QWidget()
        spacer.setFixedWidth(55)  # 设置空格的宽度
        h_layout.addWidget(spacer)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedWidth(140)
        h_layout.addWidget(line)

        line_layout.addLayout(h_layout)  # 将水平布局添加到垂直布局

        main_layout.addWidget(line_container)  # 将line_container添加到main_layout


        matrix_true = QHBoxLayout()
        matrix = QVBoxLayout()
        ok_result = QVBoxLayout()
        ok_result.addStretch(2)
        truth_label = QLabel("Truth")
        truth_label.setStyleSheet("font-size: 20px;font-weight: bold;")
        ok_result.addWidget(truth_label)
        ok_result.addStretch(1)
        matrix_true.addLayout(ok_result)

        # 添加空白空间到竖线上面
        line_container_v = QWidget()
        line_layout_v = QHBoxLayout(line_container_v)

        v_layout = QVBoxLayout()

        # 添加一个空的QWidget作为左边的空格
        spacer_v = QWidget()
        spacer_v.setFixedHeight(50)  # 设置空格的宽度
        v_layout.addWidget(spacer_v)

        line_v = QFrame()
        line_v.setFrameShape(QFrame.VLine)
        line_v.setFrameShadow(QFrame.Sunken)
        line_v.setFixedHeight(100) 
        v_layout.addWidget(line_v)

        line_layout_v.addLayout(v_layout)
        matrix_true.addWidget(line_container_v)

        # 横向布局（标签）
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("  "), alignment=Qt.AlignCenter)  # 实际结果标签
        # 创建标签并设置颜色和加粗
        ok_label = QLabel("OK")
        ok_label.setStyleSheet("font-size: 20px;color: green; font-weight: bold;")  # 绿色文字，加粗
        header_layout.addWidget(ok_label, alignment=Qt.AlignCenter)

        mid_label = QLabel("MID")
        mid_label.setStyleSheet("font-size: 20px;color: #D4AF37; font-weight: bold;")  # 黄色文字，加粗
        header_layout.addWidget(mid_label, alignment=Qt.AlignCenter)

        ng_label = QLabel("NG")
        ng_label.setStyleSheet("font-size: 20px;color: red; font-weight: bold;")  # 红色文字，加粗
        header_layout.addWidget(ng_label, alignment=Qt.AlignCenter)

        sum_label = QLabel("SUM")
        sum_label.setStyleSheet("font-size: 20px;font-weight:bold;")
        header_layout.addWidget(sum_label, alignment=Qt.AlignCenter)

        matrix.addLayout(header_layout)

        True_ok_layout = QHBoxLayout()
        # 创建标签并设置颜色和加粗
        True_ok_label = QLabel("OK")
        True_ok_label.setStyleSheet("font-size: 20px;color: green; font-weight: bold;")  # 绿色文字，加粗
        True_ok_layout.addWidget(True_ok_label, alignment=Qt.AlignCenter)

        self.ok_ok_label = QLabel("0")
        self.ok_ok_label.setStyleSheet("font-size: 20px;")
        True_ok_layout.addWidget(self.ok_ok_label, alignment=Qt.AlignCenter)

        self.ok_mid_label = QLabel("0")
        self.ok_mid_label.setStyleSheet("font-size: 20px;")
        True_ok_layout.addWidget(self.ok_mid_label, alignment=Qt.AlignCenter)

        self.ok_ng_label = QLabel("0")
        self.ok_ng_label.setStyleSheet("font-size: 20px;")
        True_ok_layout.addWidget(self.ok_ng_label, alignment=Qt.AlignCenter)

        self.ok_sum_label = QLabel("0")
        self.ok_sum_label.setStyleSheet("font-size: 20px;")
        True_ok_layout.addWidget(self.ok_sum_label, alignment=Qt.AlignCenter)

        matrix.addLayout(True_ok_layout)

        True_ng_layout = QHBoxLayout()
        # 创建标签并设置颜色和加粗
        True_ng_label = QLabel("NG")
        True_ng_label.setStyleSheet("font-size: 20px;color: red; font-weight: bold;")  # 绿色文字，加粗
        True_ng_layout.addWidget(True_ng_label, alignment=Qt.AlignCenter)

        self.ng_ok_label = QLabel("0")
        self.ng_ok_label.setStyleSheet("font-size: 20px;")
        True_ng_layout.addWidget(self.ng_ok_label, alignment=Qt.AlignCenter)

        self.ng_mid_label = QLabel("0")
        self.ng_mid_label.setStyleSheet("font-size: 20px;")
        True_ng_layout.addWidget(self.ng_mid_label, alignment=Qt.AlignCenter)

        self.ng_ng_label = QLabel("0")
        self.ng_ng_label.setStyleSheet("font-size: 20px;")
        True_ng_layout.addWidget(self.ng_ng_label, alignment=Qt.AlignCenter)

        self.ng_sum_label = QLabel("0")
        self.ng_sum_label.setStyleSheet("font-size: 20px;")
        True_ng_layout.addWidget(self.ng_sum_label, alignment=Qt.AlignCenter)

        matrix.addLayout(True_ng_layout)
        matrix.setContentsMargins(0, 0, 0, 0)
        matrix_widget = QWidget()
        matrix_widget.setLayout(matrix)
        matrix_widget.setFixedSize(250, 175)

        matrix_true.addWidget(matrix_widget)
        main_layout.addLayout(matrix_true)
        self.setLayout(main_layout)

    def update_matrix(self,ok_ok,ok_mid,ok_ng,ng_ok,ng_mid,ng_ng):
        self.ok_ok_label.setText(f"{ok_ok}")
        self.ok_mid_label.setText(f"{ok_mid}")
        self.ok_ng_label.setText(f"{ok_ng}")
        self.ok_sum_label.setText(f"{ok_ok+ok_mid+ok_ng}")
        self.ng_ok_label.setText(f"{ng_ok}")
        self.ng_mid_label.setText(f"{ng_mid}")
        self.ng_ng_label.setText(f"{ng_ng}")
        self.ng_sum_label.setText(f"{ng_ok+ng_mid+ng_ng}")

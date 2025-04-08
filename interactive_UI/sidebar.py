from PyQt5.QtWidgets import QLabel, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPolygon, QPen
from PyQt5.QtCore import Qt,QPoint,pyqtSignal

class VerticalLabel(QLabel):
    """自定义竖排显示的 QLabel"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(25, 270)  # 设置大小以适应旋转后的文字
        font = self.font()
        font.setPointSize(16)  # 设置字体大小
        self.setFont(font)

    def paintEvent(self, event):
        """绘制逆时针旋转 90° 的文字"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(-self.height() / 2, self.width() / 4, self.text())
        painter.end()

class CircularButton(QPushButton):
    """自定义圆形按钮类，带有动态切换手动绘制的箭头效果"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.arrow_direction = "<"  # 初始化为箭头指向右
        self.hovered = False  # 初始化鼠标是否在按钮上

    def paintEvent(self, event):
        """绘制圆形按钮和手动绘制的箭头"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 设置按钮的圆形背景颜色
        if self.hovered:
            pen = QPen(QColor(0,191,255), 2)  # 鼠标悬停时，外圈圆圈颜色为黑色
        else:
            pen = QPen(QColor(0, 0, 0), 2)  # 默认情况下外圈也是黑色

        painter.setPen(pen)
        painter.setBrush(QColor(255, 255, 255))  # 内部为白色
        # 绘制比按钮矩形略小的圆形，确保它在矩形范围内
        painter.drawEllipse(self.rect().adjusted(2, 2, -2, -2))

        # 绘制手动定义的箭头符号
        painter.setPen(QColor(0, 0, 0))  # 箭头为黑色
        self.draw_arrow(painter)

    def draw_arrow(self, painter):
        """手动绘制箭头"""
        rect = self.rect()
        center_x = rect.width() // 2
        center_y = rect.height() // 2

        # 箭头的大小参数
        arrow_size = min(rect.width(), rect.height()) // 4  # 控制箭头的大小
        arrow_width = arrow_size // 2  # 控制箭头的宽度，张开角度

        if self.arrow_direction == ">":
            # 定义向右的箭头坐标
            points = [
                QPoint(center_x - arrow_width, center_y - arrow_size),  # 左上
                QPoint(center_x - arrow_width, center_y + arrow_size),  # 左下
                QPoint(center_x + arrow_size, center_y),                # 右中
            ]
        else:
            # 定义向左的箭头坐标
            points = [
                QPoint(center_x + arrow_width, center_y - arrow_size),  # 右上
                QPoint(center_x + arrow_width, center_y + arrow_size),  # 右下
                QPoint(center_x - arrow_size, center_y),                # 左中
            ]

        # 使用 drawPolygon 绘制箭头
        painter.drawPolygon(QPolygon(points))

    def enterEvent(self, event):
        """鼠标进入时改变 hovered 状态"""
        self.hovered = True
        self.update()

    def leaveEvent(self, event):
        """鼠标离开时改变 hovered 状态"""
        self.hovered = False
        self.update()

    def toggle_arrow(self):
        """切换箭头方向"""
        if self.arrow_direction == ">":
            self.arrow_direction = "<"
        else:
            self.arrow_direction = ">"
        self.update()

class statistics_sidebar(QWidget):
    show_statistics_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置竖直布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(Qt.AlignTop)

        # 创建圆形按钮
        self.toggle_button = CircularButton(self)
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.clicked.connect(self.mousePressEvent)

        # 创建竖排标签
        self.label = VerticalLabel("Database Statistics", self)

        # 将按钮和标签添加到布局
        self.layout.addWidget(self.toggle_button, alignment=Qt.AlignRight)
        self.layout.addWidget(self.label, alignment=Qt.AlignRight)

        # 安装事件过滤器，检测鼠标是否进入 sidebar 区域
        self.installEventFilter(self)

    def mousePressEvent(self, event):
        if self.toggle_button.hovered :
            self.toggle_button.toggle_arrow()
            self.show_statistics_signal.emit()

    def eventFilter(self, obj, event):
        """事件过滤器用于检测 sidebar 区域的鼠标事件"""
        if obj == self:
            if event.type() == event.Enter:
                # 当鼠标进入 sidebar 区域时，将按钮设为选中状态
                self.toggle_button.enterEvent(event)
            elif event.type() == event.Leave:
                # 当鼠标离开 sidebar 区域时，取消按钮的选中状态
                self.toggle_button.leaveEvent(event)
        return super().eventFilter(obj, event)
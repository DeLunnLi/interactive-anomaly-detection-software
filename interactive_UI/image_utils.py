from PyQt5.QtWidgets import QLabel, QFrame, QSizePolicy, QMenu, QWidget, QToolBar,QAction,QDoubleSpinBox
from PyQt5.QtGui import  QPixmap, QPainter, QPen, QCursor, QPolygon, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPalette

import numpy as np
import cv2
import os

class Image(QLabel):
    """
    init signal parameter
    """
    start_annotation_signal = pyqtSignal()
    end_annotation_signal = pyqtSignal()
    save_mask_signal = pyqtSignal()
    focus_zoom_position = pyqtSignal(QPoint,int)
    focus_zoom_original_position = pyqtSignal(QPoint)
    mask_pixmap_signal = pyqtSignal(str,QPixmap)
    label_normal_signal = pyqtSignal(str,str)
    label_abnormal_signal = pyqtSignal(str,str)
    label_default_signal = pyqtSignal(str,str)

    def __init__(self):
        
        super().__init__()

        """
        init parameter

        """
        # The scale factor for the image display.
        self.current_scale_factor = 1.0

        # The parameters for drowing on the image.
        self.pen_color = Qt.white
        self.use_pen = True
        self.pen_width = 10
        self.mask_trans = 0.5
        self.drawing = False
        self.display_mask = False
        self.lines = []
        self.last_point = None
        self.image_position = QRect()

        # The parameter for display
        self.file_name = None

        # The parameter for state
        self.marker_status = 'default'  # 默认状态为default
        self.show_marker = True  # 默认显示标记
        self.show_mask = True

        """
        init function class

        """
        # Initialize QPixmap objects to store the current image and its mask
        self.current_pixmap = QPixmap()
        self.image = QPixmap()
        self.mask_pixmap = QPixmap()
        self.mask_predict = QPixmap()

        # Set basic properties of the Image Class
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.NoFrame)

    def remove_display_image(self):
        self.current_scale_factor = 1.0
        self.current_pixmap = QPixmap()
        self.mask_pixmap = QPixmap()
        self.file_name = None
        self.marker_status = 'default'  # 默认状态为default
        self.show_marker = True  # 默认不显示标记
        self.clear()

    # Function to display the image
    def display_image(self, file_name,state,image,mask):
        # 加载当前图像的标注情况（可以修改为从缩略图中获取）
        self.marker_status = state
        self.show_marker = True
        self.show_mask = True
        # Load the image from the selected file location
        self.current_pixmap = image
        self.iamge = image
        self.file_name = file_name
        
        if mask.isNull():
            # Initialize mask pixmap and set mask to be all transparent
            self.mask_pixmap = QPixmap(self.current_pixmap.size())
            self.mask_pixmap.fill(Qt.transparent)
        else:
            self.mask_pixmap = QPixmap(self.current_pixmap.size())
            self.mask_pixmap.fill(Qt.transparent)
            self.mask_predict = mask
        
        self.update_factor()
    
    # Function to caculator the max factor to make the image fill the frame
    def update_factor(self):
        if not self.current_pixmap.isNull():
            # Stretch the image to fill the entire display area
            width_scale = self.width() / self.current_pixmap.width()
            height_scale = self.height() / self.current_pixmap.height()
            self.current_scale_factor = min(width_scale, height_scale)
            self.update_displayed_image_size()
    
    def update_displayed_image_size(self):
        if self.show_mask:
            self.current_pixmap = self.mask_predict
        else:
            self.current_pixmap = self.iamge

        if not self.current_pixmap.isNull():
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.current_scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            if hasattr(self, 'mask_pixmap') and not self.mask_pixmap.isNull():
                scaled_mask = self.mask_pixmap.scaled(
                    self.mask_pixmap.size() * self.current_scale_factor,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                combined_pixmap = QPixmap(scaled_pixmap.size())
                painter = QPainter(combined_pixmap)
                painter.drawPixmap(0, 0, scaled_pixmap)
                painter.setOpacity(1 - self.mask_trans)
                painter.drawPixmap(0, 0, scaled_mask)
                
                # 叠加三角形标记
                if self.show_marker:
                    self.add_marker(painter, combined_pixmap.size())

                painter.end()
                self.setPixmap(combined_pixmap)

            else:
                final_pixmap = scaled_pixmap
                if self.show_marker:
                    combined_pixmap = QPixmap(scaled_pixmap.size())
                    painter = QPainter(combined_pixmap)
                    painter.drawPixmap(0, 0, scaled_pixmap)
                    self.add_marker(painter, scaled_pixmap.size())
                    painter.end()
                    final_pixmap = combined_pixmap
                
                self.setPixmap(final_pixmap)

            self.setFixedSize(scaled_pixmap.size())
            
            if self.drawing and self.cursor_in_image(self.mapFromGlobal(QCursor.pos())):
                self.setCursor(self.create_circle_cursor(self.pen_width * self.current_scale_factor))

    def add_marker(self, painter, size):
        if self.marker_status == 'default':
            return
        # 在右上角绘制三角形
        triangle_size = int(0.1 * self.current_pixmap.width() * self.current_scale_factor)  # 控制三角形大小并确保为整数

        # 使用 QPolygon 创建三角形
        points = QPolygon([
            QPoint(size.width() - triangle_size, 0),
            QPoint(size.width(), 0),
            QPoint(size.width(), triangle_size)
        ])

        if self.marker_status == 'normal':
            color = QColor(Qt.green)# 正常状态为深绿色
        elif self.marker_status == 'abnormal':
            color = QColor(Qt.red) # 异常状态为深红色
        painter.setOpacity(1)
        # 设置笔刷并绘制三角形
        painter.setBrush(color)
        painter.drawPolygon(points)

    # Function to handle the mouse wheel event
    def wheelEvent(self, event):
        self.trigger_focus_original_position(event.pos())
        if event.angleDelta().y() > 0:
            # Zoom in
            self.current_scale_factor *= 1.1
            self.update_displayed_image_size() 
            self.trigger_focus_zoom(event.pos(),1.1)
        else:
            # Zoom out
            self.current_scale_factor /= 1.1  
            self.update_displayed_image_size() 
            self.trigger_focus_zoom(event.pos(),1.0/1.1)

    # def wheelEvent(self, event):
    #     # 获取鼠标在控件中的位置
    #     mouse_pos_1 = event.pos()
    #     width = self.width()
    #     height = self.height()
    #     # 获取缩放之前的图像尺寸
    #     old_width = self.width() * self.current_scale_factor
    #     old_height = self.height() * self.current_scale_factor

    #     # 计算鼠标在缩放前图像中的相对坐标
    #     mouse_rel_x = mouse_pos_1.x() / old_width  # X方向相对位置
    #     mouse_rel_y = mouse_pos_1.y() / old_height # Y方向相对位置

    #     # 根据滚轮方向调整缩放比例
    #     if event.angleDelta().y() > 0:
    #         # Zoom in
    #         self.current_scale_factor *= 1.1  
    #     else:
    #         # Zoom out
    #         self.current_scale_factor /= 1.1  

    #     # 更新图像显示大小
    #     self.update_displayed_image_size()

    #     mouse_pos_2 = event.pos()

    #     # 获取缩放之后的图像尺寸
    #     new_width = self.width() * self.current_scale_factor
    #     new_height = self.height() * self.current_scale_factor

    #     # 计算缩放后鼠标相对图像左上角的坐标（根据缩放比例更新）
    #     new_mouse_x = mouse_rel_x * new_width  # 缩放后鼠标在图像中的X坐标
    #     new_mouse_y = mouse_rel_y * new_height # 缩放后鼠标在图像中的Y坐标

    #     # 打印信息
    #     print(f"Mouse Position1 in Widget: {mouse_pos_1}")
    #     print(f"Mouse Position2 in Widget: {mouse_pos_2}")
    #     print(f"Mouse Position in Image: ({new_mouse_x:.2f}, {new_mouse_y:.2f})")
    #     print(f"Current Scale Factor: {self.current_scale_factor:.2f}")
    #     print(f"Old Image Size: {old_width:.2f} x {old_height:.2f}")
    #     print(f"New Image Size: {new_width:.2f} x {new_height:.2f}")
    #     print(f"Original Image Size: {self.width()} x {self.height()}")
    #     print(f"Original Image Size: {width} x {height}")

    # Function to handle the mouse press event
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            # Record the point of mouse leftbutton click
            self.last_point = event.pos()

    # Function to handle the mouse move event and record the path.
    def mouseMoveEvent(self, event):
        # When in drawing model and cursor in the range of image, change the curosr to circular shape.
        if self.drawing and self.cursor_in_image(event.pos()):
            self.setCursor(self.create_circle_cursor(self.pen_width*self.current_scale_factor))
        else:
            self.unsetCursor()

        # Update last point and record the path of mouse movements in lines
        if self.last_point is not None: 
            new_pos = event.pos()
            self.lines.append((self.last_point, new_pos))
            self.last_point = new_pos
            self.update()

    # Function to handle the mouse releaseevent
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = None

    # Function to paint the path of mouse movements on the mask
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.current_pixmap and not self.current_pixmap.isNull():
            # Map the mouse position to the corresponding area in the mask
            image_offset_point = self.image_offset_point()
            mask_painter = QPainter(self.mask_pixmap)
            # Draw the path on mask
            if self.use_pen:
                mask_painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap)) # White for mask
            else:
                mask_painter.setCompositionMode(QPainter.CompositionMode_Clear)
                mask_painter.setPen(QPen(Qt.transparent, self.pen_width, Qt.SolidLine, Qt.RoundCap))
            for line in self.lines:
                start, end = line
                start = (start- image_offset_point) / self.current_scale_factor 
                end = (end - image_offset_point) / self.current_scale_factor 
                mask_painter.drawLine(start, end)
            mask_painter.end()

            # update the mask
            self.update_displayed_image_size()
            self.lines = []

    # Function to caculate image offset point
    def image_offset_point(self):
        image_rect = self.pixmap().rect()
        image_x_offset = (self.width() - image_rect.width())/2
        image_y_offset = (self.height() - image_rect.height())/2
        image_offset_point = QPoint(image_x_offset, image_y_offset)
        return image_offset_point

    # Function to translate the Qpixmap into cv2's mask type
    def save_image_with_drawing(self):
        self.mask_pixmap_signal.emit(self.file_name,self.mask_pixmap)
        print(self.file_name)
        mask_image = self.mask_pixmap.toImage()

        # Use mask_array store the mask
        width = mask_image.width()
        height = mask_image.height()
        mask_array = np.zeros((height, width), dtype=np.uint8)
        # Converse pixel-by-pixel
        for y in range(height):
            for x in range(width):
                pixel_color = mask_image.pixelColor(x, y)
                gray_value = pixel_color.value()
                mask_array[y, x] = 255 if gray_value > 128 else 0

        # Get the image location
        # Seprater the image location path
        normalized_path = os.path.normpath(self.file_name)
        parts = normalized_path.split(os.sep)
        relative_path = os.path.join(parts[-4], parts[-3], parts[-2])
        # Set the mask folder 
        target_dir = os.path.join("cache", "mask", relative_path)
        os.makedirs(target_dir, exist_ok=True)
        file_name = parts[-1]
        # Set mask name and path
        mask_file_name = file_name.replace(".png","_mask.png")
        mask_file_path = os.path.join(target_dir, mask_file_name)
        # Use cv2 store the mask image
        cv2.imwrite(mask_file_path, mask_array)

    def delete_image_mask(self):
        normalized_path = os.path.normpath(self.file_name)
        parts = normalized_path.split(os.sep)
        relative_path = os.path.join(parts[-4], parts[-3], parts[-2])
        # Set the mask folder 
        target_dir = os.path.join("cache", "mask", relative_path)
        os.makedirs(target_dir, exist_ok=True)
        file_name = parts[-1]
        # Set mask name and path
        mask_file_name = file_name.replace(".png","_mask.png")
        mask_file_path = os.path.join(target_dir, mask_file_name)
        if os.path.exists(mask_file_path):
            os.remove(mask_file_path)

    # Function to create the circle pen curosr
    def create_circle_cursor(self, diameter):
        # Create the circle according to the size of celected pen
        pixmap = QPixmap(diameter + 8, diameter + 8)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawEllipse(4, 4, diameter, diameter)
        painter.end()
        # Set the hot location in center of the circlel
        cursor = QCursor(pixmap, diameter // 2 + 4, diameter // 2 + 4)
        return cursor
    
    # Function to judge the location of cursor
    def cursor_in_image(self,position):
        if self.get_image_position().contains(position):
            return True
        else:
            return False
        
    # Function to caculate the location that image displayed
    def get_image_position(self):
        if self.pixmap() is None:
            return None
        else:
            image_x_min = (self.width() - self.pixmap().width()) / 2
            image_y_min = (self.height() - self.pixmap().height()) / 2
            image_width = self.pixmap().width()
            image_height = self.pixmap().height()
            return QRect(image_x_min, image_y_min, image_width, image_height)
    
    # Function to handle the leftbutton press event.
    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        label_as_normal_sample = context_menu.addAction("Label as Normal Sample")
        label_as_abnormal_sample = context_menu.addAction("Label as Abnormal Sample")
        label_as_default_label = context_menu.addAction("Remove Label")

        context_menu.addSeparator()
        hide_label = context_menu.addAction("Hide Label")
        show_label = context_menu.addAction("Show Label")

        context_menu.addSeparator()
        start_annotation = context_menu.addAction("Edit Defect Area")
        end_annotation = context_menu.addAction("Exit Edit Mode")
        apply_annotation = context_menu.addAction("Set Edited Area as Mask")
        remove_annotation = context_menu.addAction("Remove Edited Mask and Label")

        context_menu.addSeparator()
        hide_mask = context_menu.addAction("Hide Predict Mask")
        show_mask = context_menu.addAction("Show Predict Mask")



        start_annotation.triggered.connect(self.trigger_start_annotation)
        end_annotation.triggered.connect(self.trigger_end_annotation)
        apply_annotation.triggered.connect(self.trigger_apply_annotation)
        remove_annotation.triggered.connect(self.trigger_remove_annotation)
        label_as_normal_sample.triggered.connect(self.trigger_normal_sample)
        label_as_abnormal_sample.triggered.connect(self.trigger_abnormal_sample)
        label_as_default_label.triggered.connect(self.trigger_default_sample)
        hide_label.triggered.connect(self.hide_image_label)
        show_label.triggered.connect(self.show_image_label)
        hide_mask.triggered.connect(self.hide_image_mask)
        show_mask.triggered.connect(self.show_image_mask)
        context_menu.exec_(event.globalPos())

    """
    Functions to handle the marker state
    """
    def trigger_normal_sample(self):
        self.marker_status = 'normal'
        self.update_displayed_image_size()
        self.label_normal_signal.emit(self.file_name,'normal')

    def trigger_abnormal_sample(self):
        self.marker_status = 'abnormal'
        self.update_displayed_image_size()
        self.label_abnormal_signal.emit(self.file_name,'abnormal')

    def trigger_default_sample(self):
        self.marker_status = 'default'
        self.update_displayed_image_size()
        self.label_default_signal.emit(self.file_name,'default')

    def hide_image_label(self):
        self.show_marker = False
        self.update_displayed_image_size()
    
    def show_image_label(self):
        self.show_marker = True
        self.update_displayed_image_size()

    def hide_image_mask(self):
        self.show_mask = False
        self.update_displayed_image_size()
    
    def show_image_mask(self):
        self.show_mask = True
        self.update_displayed_image_size()


    """
    Functions to handle the focus zoom
    """
    def trigger_focus_zoom(self,position,factor):
        self.focus_zoom_position.emit(position,factor)

    def trigger_focus_original_position(self,position):
        self.focus_zoom_original_position.emit(position)

    """
    Functions to handle the menu method and send signal to homepage 
    """
    def trigger_start_annotation(self):
        self.drawing = True
        if  self.pixmap() is not None:
            self.start_annotation_signal.emit()

    def trigger_end_annotation(self):
        self.drawing = False
        if  self.pixmap() is not None:
            self.end_annotation_signal.emit()

    def trigger_apply_annotation(self):
        self.marker_status = 'abnormal'
        self.update_displayed_image_size()
        if  self.pixmap() is not None:
            self.save_image_with_drawing()

    def trigger_remove_annotation(self):
        mask_pixmap = QPixmap(self.current_pixmap.size())
        mask_pixmap.fill(Qt.transparent)
        self.mask_pixmap = mask_pixmap
        self.trigger_default_sample()
        self.delete_image_mask()
        self.update_displayed_image_size()
        
    """
    Functions to handle the homepage toolbar's selection event, renew the set of the qpen 
    """
    def change_pen_mode(self,color):
        self.pen_color = color

    def change_pen_width(self,width):
        self.pen_width = width

    def change_mask_trans(self,trans):
        self.mask_trans = trans/100
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QFileDialog, QMenu, QAction,QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPolygon, QColor,QFont,QImage
from PyQt5.QtCore import Qt, pyqtSignal,QSize,QPoint,QRect
from skimage import morphology,measure
from PIL import Image
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.draw import polygon_perimeter
from skimage.segmentation import mark_boundaries,find_boundaries
import cv2
import sip
import os
import time
import numpy as np

from anomaly_detection_algorithm.PaDiM.unspervised_algorithm_PaDiM import PaDiM

class ThumbnailManager(QWidget):
    thumbnail_selected = pyqtSignal(str,str,QPixmap,QPixmap)
    image_num = pyqtSignal(int)
    labeled_num = pyqtSignal(int)
    trained_num = pyqtSignal(int)
    average_time = pyqtSignal(float)
    labeled_scores = pyqtSignal(np.ndarray,np.ndarray)
    matrix = pyqtSignal(int,int,int,int,int,int)
    remove_display_image = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(0)
        
        # 创建文字标签
        self.filter_label = QLabel("Display Filter ")
        self.filter_label.setContentsMargins(0, 0, 0, 0)  # 去除内部边距
        filter_layout.addWidget(self.filter_label)

        # 创建筛选框
        self.filter_combobox = QComboBox()
        self.filter_combobox.setContentsMargins(0, 0, 0, 0)  # 去除内部边距
        self.filter_combobox.addItem("All")
        self.filter_combobox.addItem("New")
        self.filter_combobox.addItem("Labeled")
        self.filter_combobox.addItem("Unlabeled")
        self.filter_combobox.addItem("Trained")
        self.filter_combobox.addItem("Untrained")
        self.filter_combobox.addItem("Normal")
        self.filter_combobox.addItem("Abnormal")
        self.filter_combobox.currentTextChanged.connect(self.filter_thumbnails_by_state)

        filter_layout.addWidget(self.filter_combobox)
        
        # 设置伸展因子：label的伸展因子为0，combobox的伸展因子为1
        filter_layout.setStretch(0, 0)  # QLabel占用其最小空间
        filter_layout.setStretch(1, 1)  # QComboBox填充剩余空间

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QVBoxLayout()
        
        # 将布局设置为靠左对齐
        self.thumbnail_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(2)

        self.visible_state = 'All'
        self.thumbnail_widget.setLayout(self.thumbnail_layout)
        self.scroll_area.setWidget(self.thumbnail_widget)
        self.image_files = []
        self.image_files_new = []
        self.thumbnails = {}  # 用于存储缩略图及其对应的图像
        self.current_file_name = None
        self.selected_thumbnail = None  # 记录当前选中的缩略图

        self.model = None
        self.need_process_score = False

        self.mask1 = None
        self.mask2 = None

        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(filter_layout)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)


    def update_thumbnails(self, new_files):
        # Append new files to the existing image list
        self.image_files.extend(new_files)
        # Update layout with new thumbnails
        for file_name in new_files:
            pixmap = QPixmap(file_name)
            thumbnail = QLabel()
            scaled_pixmap = self.scale_pixmap(pixmap, 100)
            thumbnail.setPixmap(scaled_pixmap)
            mask_pixmap = QPixmap(pixmap.size())
            mask_pixmap.fill(Qt.transparent)
            # 调整缩略图的对齐方式为左对齐
            thumbnail.setAlignment(Qt.AlignLeft)

            # 设置默认样式
            thumbnail.setStyleSheet("border: 2px solid transparent;")  # 默认没有边框

            # Set the event handler for mouse press, including right click
            thumbnail.mousePressEvent = lambda event, f=file_name, t=thumbnail: self.handle_thumbnail_selected(event, f, t)

            
            self.thumbnails[file_name] = {
                "widget": thumbnail, 
                "image": pixmap, 
                "mask": mask_pixmap, 
                "masked_image": pixmap,
                "heatmap": None,
                "heatmap_image": pixmap,
                "selected_heatmap":mask_pixmap,
                "circule_pixmap":mask_pixmap,
                "show_pixmap":pixmap,
                "label_state": "default", 
                "mask_state":"default",
                "trained":False,
                "distinguish_state":"default",
                "show_state":"default",
                "visible_state": True,
                "score": 0,
                "score_map": None,
                "mask_heatmap_image":None
                }
            self.thumbnail_layout.addWidget(thumbnail)
        self.adjust_thumbnail_size()
        self.image_num.emit(len(self.image_files))


    def handle_thumbnail_selected(self, event, file_name, thumbnail):
        # 判断是否是鼠标左键
        if event.button() == Qt.LeftButton:

            # 取消之前选择的缩略图的样式
            if self.selected_thumbnail is not None and not sip.isdeleted(self.selected_thumbnail):
                self.selected_thumbnail.setStyleSheet("border: 2px solid transparent;")

            # 更新新的选择
            self.selected_thumbnail = thumbnail
            self.selected_thumbnail.setStyleSheet("border: 2px solid blue;")  # 新的样式
        
            # 如果已经有选中的缩略图，先清除其选中样式
            if self.selected_thumbnail:
                self.selected_thumbnail.setStyleSheet("border: 2px solid transparent;")

            # 设置当前缩略图为选中状态，边框加粗
            thumbnail.setStyleSheet("border: 2px solid blue;")
            self.selected_thumbnail = thumbnail  # 更新选中的缩略图

            # 触发信号，传递各种属性
            state = self.thumbnails[file_name]["label_state"]
            # image = self.thumbnails[file_name]["image"]

            image = self.thumbnails[file_name]["image"]
            mask = self.thumbnails[file_name]["show_pixmap"]
            self.current_file_name = file_name
            self.thumbnail_selected.emit(file_name,state,image,mask)
        elif event.button() == Qt.RightButton:
            # 如果是右键，则显示上下文菜单
            self.show_context_menu(event.globalPos(), file_name)

    def update_selected(self):
        if self.current_file_name != None:
            file_name = self.current_file_name
            state = self.thumbnails[file_name]["label_state"]
            image = self.thumbnails[file_name]["image"]
            mask = self.thumbnails[file_name]["show_pixmap"]
            self.thumbnail_selected.emit(file_name,state,image,mask)


    def show_context_menu(self, position, file_name):
        # 创建上下文菜单
        context_menu = QMenu(self)

        # 添加菜单项
        delete_action = QAction("Delete", self)
        open_action = QAction("Open", self)

        # 将菜单项连接到槽函数
        delete_action.triggered.connect(lambda: self.delete_thumbnail(file_name))
        open_action.triggered.connect(lambda: self.open_image(file_name))

        # 将菜单项添加到菜单中
        context_menu.addAction(open_action)
        context_menu.addAction(delete_action)

        # 在指定位置显示菜单
        context_menu.exec_(position)

    def filter_thumbnails_by_state(self, state):
        self.visible_state = state
        # 通过状态筛选显示的缩略图
        if state == 'All':
            for file_name, data in self.thumbnails.items():
                data["visible_state"] = True
        elif state == 'New':
            for file_name, data in self.thumbnails.items():
                if file_name in self.image_files_new:
                    data["visible_state"] = True
                else:
                    data["visible_state"] = False
        elif state == 'Labeled':
            for file_name, data in self.thumbnails.items():
                if data["label_state"] in ['normal', 'abnormal']:
                    data["visible_state"] = True
                else:
                    data["visible_state"] = False
        elif state == 'Unlabeled':
            for file_name, data in self.thumbnails.items():
                if data["label_state"] in ['normal', 'abnormal']:
                    data["visible_state"] = False
                else:
                    data["visible_state"] = True
        elif state == 'Trained':
            for file_name, data in self.thumbnails.items():
                if data["trained"]:
                    data["visible_state"] = True
                else:
                    data["visible_state"] = False
        elif state == 'Untrained':
            for file_name, data in self.thumbnails.items():
                if data["trained"]:
                    data["visible_state"] = False
                else:
                    data["visible_state"] = True    
        elif state == 'Normal':
            for file_name, data in self.thumbnails.items():
                if data["label_state"] == 'normal':
                    data["visible_state"] = True
                else:
                    data["visible_state"] = False
        elif state == 'Abnormal':
            for file_name, data in self.thumbnails.items():
                if data["label_state"] == 'abnormal':
                    data["visible_state"] = True
                else:
                    data["visible_state"] = False
        self.adjust_thumbnail_size()

    def delete_thumbnail(self, file_name):
        # 检查字典中是否存在对应的文件名
        if file_name in self.thumbnails:
            if file_name == self.current_file_name:
                self.remove_display_image.emit()
                self.current_file_name = None
            # 获取缩略图的 QLabel widget
            thumbnail = self.thumbnails[file_name]["widget"]
            # 从布局中移除缩略图
            self.thumbnail_layout.removeWidget(thumbnail)
            # 删除 QLabel widget
            thumbnail.deleteLater()
            # 从字典中删除缩略图信息
            del self.thumbnails[file_name]
            # 从文件列表中移除相应的文件名
            if file_name in self.image_files:
                self.image_files.remove(file_name)

    def remove_selected_images(self):
        for file_name, data in list(self.thumbnails.items()):
            if data["visible_state"] == True:
                self.delete_thumbnail(file_name)
        self.image_num.emit(len(self.image_files))
        self.emit_labeled_count()

    def open_image(self, file_name):
        # 打开图像，可以实现显示原始图像的功能
        print(f"Open image: {file_name}")

    def adjust_thumbnail_size(self):
        # 调整缩略图大小
        scroll_area_width = self.scroll_area.width() - 28
        for file_name, data in self.thumbnails.items():
            if data["visible_state"] == True:
                data["widget"].show()
                thumbnail = data["widget"]
                if data["masked_image"].isNull() and data["heatmap_image"].isNull():
                    pixmap = data["image"]
                elif data["heatmap_image"].isNull():
                    pixmap = data["masked_image"]
                else:
                    # 可能需要改回
                    # pixmap = data["heatmap_image"]
                    pixmap = data["show_pixmap"]
                scaled_pixmap = self.scale_pixmap(pixmap, scroll_area_width)
                if data["label_state"] != 'default':
                    painter = QPainter(scaled_pixmap)
                    self.add_marker(file_name,painter,scaled_pixmap.size())
                thumbnail.setPixmap(scaled_pixmap)
            else:
                data["widget"].hide()

    def scale_pixmap(self, pixmap, max_size):
        # 按比例缩放图片
        original_size = pixmap.size()
        aspect_ratio = original_size.height() / original_size.width()
        if original_size.height() > original_size.width():
            new_height = max_size
            new_width = int(new_height / aspect_ratio)
        else:
            new_width = max_size
            new_height = int(new_width * aspect_ratio)
    
        return pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    def load_images(self):
        self.image_files_new = []
        # 打开文件对话框以选择多个图像
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.xpm *.jpg)")
        self.image_files_new = file_names
        if file_names:
            # 使用新文件更新缩略图
            self.update_thumbnails(file_names)
    

    def add_thumbnail_mask(self, file_name, mask_pixmap):
        """
        接收掩码并更新相应的缩略图显示。
        :param file_name: 对应的图像文件名
        :param mask_pixmap: 掩码的 QPixmap
        """
        if file_name in self.thumbnails:
            thumbnail_data = self.thumbnails[file_name]
            # 更新掩码
            thumbnail_data["mask"] = mask_pixmap

            # 更新显示的缩略图，合并掩码
            self.update_thumbnail_display(file_name)

    def update_thumbnail_display(self, file_name):
        """
        更新缩略图显示，将掩码应用到缩略图上。
        :param file_name: 对应的图像文件名
        """
        if file_name in self.thumbnails:
            thumbnail_data = self.thumbnails[file_name]
            original_pixmap = thumbnail_data["image"]
            mask_pixmap = thumbnail_data["mask"]

            if mask_pixmap:
                # 创建合成图像
                combined_pixmap = self.create_combined_pixmap(original_pixmap, mask_pixmap)
                scroll_area_width = self.scroll_area.width() - 28
                scale_combined_pixmap = self.scale_pixmap(combined_pixmap,scroll_area_width)
                thumbnail_data["widget"].setPixmap(scale_combined_pixmap)
                thumbnail_data["masked_image"] = combined_pixmap

            self.thumbnails[file_name]["label_state"] = "abnormal"
            self.adjust_thumbnail_size()

    
    def create_combined_pixmap(self, original_pixmap, mask_pixmap):
        """
        合成原始图像和掩码图像，并添加得分框。
        :param original_pixmap: 原始图像 QPixmap
        :param mask_pixmap: 掩码 QPixmap
        :param score: 图像得分
        :return: 合成后的 QPixmap
        """
        # 将掩码图像调整到与原始图像相同的大小
        mask_pixmap = mask_pixmap.scaled(original_pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 创建一个透明的 QPixmap
        combined_pixmap = QPixmap(original_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        # 使用 QPainter 绘制合成图像
        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, original_pixmap)  # 绘制原始图像
        painter.setOpacity(0.5) 
        painter.drawPixmap(0, 0, mask_pixmap)      # 绘制掩码图像

        painter.end()

        return combined_pixmap

    def create_combined_pixmap_score(self, original_pixmap, mask_pixmap, score):
        """
        合成原始图像和掩码图像，并添加得分框。
        :param original_pixmap: 原始图像 QPixmap
        :param mask_pixmap: 掩码 QPixmap
        :param score: 图像得分
        :return: 合成后的 QPixmap
        """
        # 将掩码图像调整到与原始图像相同的大小
        mask_pixmap = mask_pixmap.scaled(original_pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 创建一个透明的 QPixmap
        combined_pixmap = QPixmap(original_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        # 使用 QPainter 绘制合成图像
        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, original_pixmap)  # 绘制原始图像
        painter.setOpacity(0.5) 
        painter.drawPixmap(0, 0, mask_pixmap)      # 绘制掩码图像

        # 添加得分框
        score_rect = QRect(0, original_pixmap.height() - 100, 240, 100)  # 设定长方形的位置和大小
        painter.setBrush(QColor(0, 255, 0))  # 设置背景色为白色
        painter.drawRect(score_rect)  # 绘制长方形

        # 添加得分文本，保留小数点后三位
        painter.setPen(Qt.black)  # 设置文本颜色为黑色
        painter.setFont(QFont("Arial", 48, QFont.Bold))  # 设置字体和大小
        painter.drawText(score_rect, Qt.AlignCenter, f"{score:.3f}")  # 在长方形中居中绘制得分

        painter.end()

        return combined_pixmap
    
    def create_combined_pixmap_score_without_mask(self, original_pixmap, score):
        """
        合成原始图像和掩码图像，并添加得分框。
        :param original_pixmap: 原始图像 QPixmap
        :param mask_pixmap: 掩码 QPixmap
        :param score: 图像得分
        :return: 合成后的 QPixmap
        """

        # 创建一个透明的 QPixmap
        combined_pixmap = QPixmap(original_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        # 使用 QPainter 绘制合成图像
        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, original_pixmap)  # 绘制原始图像
        painter.setOpacity(0.5) 

        # 添加得分框
        score_rect = QRect(0, original_pixmap.height() - 100, 240, 100)  # 设定长方形的位置和大小
        painter.setBrush(QColor(0, 255, 0))  # 设置背景色为白色
        painter.drawRect(score_rect)  # 绘制长方形

        # 添加得分文本，保留小数点后三位
        painter.setPen(Qt.black)  # 设置文本颜色为黑色
        painter.setFont(QFont("Arial", 48, QFont.Bold))  # 设置字体和大小
        painter.drawText(score_rect, Qt.AlignCenter, f"{score:.3f}")  # 在长方形中居中绘制得分

        painter.end()

        return combined_pixmap

    
    # 此函数与image_utils中重复，可独立为一个绘制函数
    def add_marker(self, file_name, painter, size):
        state = self.thumbnails[file_name]["label_state"]
        if state == 'default':
            return
        # 在右上角绘制三角形
        triangle_size = int(0.1 * size.width())  # 控制三角形大小并确保为整数

        # 使用 QPolygon 创建三角形
        points = QPolygon([
            QPoint(size.width() - triangle_size, 0),
            QPoint(size.width(), 0),
            QPoint(size.width(), triangle_size)
        ])

        if state == 'normal':
            color = QColor(Qt.green)# 正常状态为深绿色
        elif state == 'abnormal':
            color = QColor(Qt.red) # 异常状态为深红色
        painter.setOpacity(1)
        # 设置笔刷并绘制三角形
        painter.setBrush(color)
        painter.drawPolygon(points)
        painter.end()

    def update_display_image_label(self,file_name,label):
        if label == 'normal':
            self.thumbnails[file_name]["mask"].fill(Qt.transparent)
            self.thumbnails[file_name]["mask_image"] = self.thumbnails[file_name]["image"]
        self.thumbnails[file_name]["label_state"] = label
        # 此处暂时使用此函数，可以仅更新修改内容以节约时间
        self.export_scores()
        self.emit_labeled_count()
        self.adjust_thumbnail_size()

    def label_selected_image_as_normal(self):
        for file_name, data in self.thumbnails.items():
            if data["visible_state"] == True:
                data["label_state"] = "normal"
        self.adjust_thumbnail_size()
        self.emit_labeled_count()
        self.export_scores()
        if not self.current_file_name==None:
            self.update_display_image()

    def label_selected_image_as_abnormal(self):
        for file_name, data in self.thumbnails.items():
            if data["visible_state"] == True:
                data["label_state"] = "abnormal"
        self.adjust_thumbnail_size()
        self.emit_labeled_count()
        self.export_scores()
        if not self.current_file_name==None:
            self.update_display_image()

    def emit_labeled_count(self):
        labeled_count = 0 
        for file_name, data in self.thumbnails.items():
            if data["label_state"]!="default":
                labeled_count+=1
        self.labeled_num.emit(labeled_count)

    def update_display_image(self):
        state = self.thumbnails[self.current_file_name]["label_state"]
        image = self.thumbnails[self.current_file_name]["image"]
        mask = self.thumbnails[self.current_file_name]["mask"]
        self.thumbnail_selected.emit(self.current_file_name,state,image,mask)

    def train_model(self):
        model = PaDiM()
        normal_file_name = []
        for file_name, data in self.thumbnails.items():
            if data["label_state"] == "normal":
                data["trained"] = True
                normal_file_name.append(file_name)
        model.train(normal_file_name)
        self.trained_num.emit(len(normal_file_name))
        self.model = model
    
    def process_images(self):
        self.need_process_score = True
        need_process_image = []
        for file_name,data in self.thumbnails.items():
            need_process_image.append(file_name)

        
        start_time = time.time()
        img_score,scores = self.model.test(need_process_image)
        end_time = time.time()
        process_time = (end_time - start_time) * 1000
        average_process_time = process_time/len(need_process_image)
        self.average_time.emit(average_process_time)
        
        normal_scores = []
        abnormal_scores = []
        for index, (file_name, data) in enumerate(self.thumbnails.items()):

            heatmap_path = caculate_heatmap_path(file_name)
            heatmap = QPixmap(heatmap_path)
            self.thumbnails[file_name]["score"] = img_score[index]
            self.thumbnails[file_name]["heatmap"] = heatmap
            self.thumbnails[file_name]["heatmap_image"] = self.create_combined_pixmap_score(self.thumbnails[file_name]["image"],heatmap,img_score[index])
            self.thumbnails[file_name]["score_map"] = scores[index]
            if self.thumbnails[file_name]["label_state"] =="normal":
                normal_scores.append(img_score[index])
            elif self.thumbnails[file_name]["label_state"] =="abnormal":
                abnormal_scores.append(img_score[index])

        normal_scores_np = np.array(normal_scores)
        abnormal_scores_np = np.array(abnormal_scores)

        self.labeled_scores.emit(normal_scores_np,abnormal_scores_np)

        self.adjust_thumbnail_size()

    def create_mask_heatmap_pixmap(self,mask1_thrshold,mask2_thrshold):
        kernel = morphology.disk(4)
        ok_ok = 0
        ok_mid = 0
        ok_ng = 0
        ng_ok = 0
        ng_mid = 0
        ng_ng = 0
        for file_name,data in self.thumbnails.items():
            if self.thumbnails[file_name]["score"] <= mask1_thrshold:
                if self.thumbnails[file_name]["label_state"] == 'normal':
                    ok_ok +=1
                elif self.thumbnails[file_name]["label_state"] == 'abnormal':
                    ng_ok+=1
                if self.mask1 is None:
                    self.thumbnails[file_name]["show_pixmap"] = self.create_combined_pixmap_score_without_mask(self.thumbnails[file_name]["image"],self.thumbnails[file_name]["score"])
                    continue
                elif self.mask1 <= mask1_thrshold:
                    if self.thumbnails[file_name]["score"] <= self.mask1:
                        continue
                    else:
                        self.thumbnails[file_name]["show_pixmap"] = self.create_combined_pixmap_score_without_mask(self.thumbnails[file_name]["image"],self.thumbnails[file_name]["score"])
                        continue
                elif self.mask1 > mask1_thrshold:
                    if self.thumbnails[file_name]["score"] <= mask1_thrshold:
                        continue
                    else:
                        self.thumbnails[file_name]["show_pixmap"] = self.create_combined_pixmap_score_without_mask(self.thumbnails[file_name]["image"],self.thumbnails[file_name]["score"])
                        continue
            heatmap_path = caculate_heatmap_path(file_name)
            heatmap_image = Image.open(heatmap_path)
            rgb_image = heatmap_image.convert('RGB')
            heatmap = np.array(rgb_image)

            if self.thumbnails[file_name]["label_state"] == 'normal' and self.thumbnails[file_name]["score"] > mask1_thrshold and self.thumbnails[file_name]["score"] <= mask2_thrshold:
                ok_mid +=1
            elif self.thumbnails[file_name]["label_state"] == 'normal' and self.thumbnails[file_name]["score"] > mask2_thrshold:
                ok_ng+=1
            elif self.thumbnails[file_name]["label_state"] == 'abnormal' and self.thumbnails[file_name]["score"] > mask1_thrshold and self.thumbnails[file_name]["score"] <= mask2_thrshold:
                ng_mid+=1
            elif self.thumbnails[file_name]["label_state"] == 'abnormal' and self.thumbnails[file_name]["score"] > mask2_thrshold:
                ng_ng+=1

            if self.mask1 is None and self.mask2 is None:
                self.create_selected_heatmap(file_name,mask1_thrshold,kernel,heatmap)
                self.create_circle_pixmap(file_name,mask2_thrshold,kernel,heatmap)
            elif self.mask1 == mask1_thrshold:
                self.create_circle_pixmap(file_name,mask2_thrshold,kernel,heatmap)
            elif self.mask2 == mask2_thrshold:
                self.create_selected_heatmap(file_name,mask1_thrshold,kernel,heatmap)
            else:
                self.create_selected_heatmap(file_name,mask1_thrshold,kernel,heatmap)
                self.create_circle_pixmap(file_name,mask2_thrshold,kernel,heatmap)

            self.create_circule_pixmap_score(file_name,mask2_thrshold)

        self.adjust_thumbnail_size()

        self.mask1 = mask1_thrshold
        self.mask2 = mask2_thrshold
        self.matrix.emit(ok_ok,ok_mid,ok_ng,ng_ok,ng_mid,ng_ng)

        self.update_selected()


    def create_selected_heatmap(self,file_name,thrshold,kernel,heatmap):
        mask = self.thumbnails[file_name]["score_map"].copy()
        mask[mask>thrshold] = 1
        mask[mask<=thrshold] = 0
        mask = morphology.opening(mask, kernel)
        heatmap_filtered = heatmap * np.expand_dims(mask, axis=-1)
        heatmap_filtered_pixmap = numpy_array_to_qpixmap_with_transparency(heatmap_filtered)
        self.thumbnails[file_name]["selected_heatmap"] = heatmap_filtered_pixmap

    def create_circle_pixmap(self,file_name,thrshold,kernel,heatmap):
        mask = self.thumbnails[file_name]["score_map"].copy()
        mask[mask>thrshold] = 1
        mask[mask<=thrshold] = 0
        mask = morphology.opening(mask, kernel)
        mask*=255
        boundaries_mask = find_boundaries(mask)
        circule_array = np.zeros_like(heatmap)
        if boundaries_mask.any():
            circule = mark_boundaries(circule_array, mask, color=(255, 0, 0), mode='thick')
        else:
            circule = circule_array
        circule_pixmap = numpy_array_to_qpixmap_with_transparency(circule)
        self.thumbnails[file_name]["circule_pixmap"] = circule_pixmap


################
    def create_circule_pixmap_score(self, file_name,mask):
        thumbnail_data = self.thumbnails[file_name]
        original_pixmap = thumbnail_data["image"]
        score = thumbnail_data["score"]
        heatmap_pixmap = thumbnail_data["selected_heatmap"]
        circule_pixmap = thumbnail_data["circule_pixmap"]
        # 将掩码图像调整到与原始图像相同的大小
        heatmap_pixmap = heatmap_pixmap.scaled(original_pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        circule_pixmap = circule_pixmap.scaled(original_pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 创建一个透明的 QPixmap
        combined_pixmap = QPixmap(original_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        # 使用 QPainter 绘制合成图像
        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, original_pixmap)  # 绘制原始图像
        painter.setOpacity(0.5) 
        painter.drawPixmap(0, 0, heatmap_pixmap)      # 绘制掩码图像
        painter.setOpacity(1) 
        painter.drawPixmap(0, 0, circule_pixmap)      # 绘制掩码图像
        painter.setOpacity(0.5) 
        # 添加得分框
        score_rect = QRect(0, original_pixmap.height() - 100, 240, 100)  # 设定长方形的位置和大小
        if score<=mask:
            color = QColor(128,128,128)
        else:
            color = QColor(255,0,0)
        painter.setBrush(color)
        painter.drawRect(score_rect)

        # 添加得分文本，保留小数点后三位
        painter.setPen(Qt.black)  # 设置文本颜色为黑色
        painter.setFont(QFont("Arial", 48, QFont.Bold))  # 设置字体和大小
        painter.drawText(score_rect, Qt.AlignCenter, f"{score:.3f}")  # 在长方形中居中绘制得分

        painter.end()

        thumbnail_data["show_pixmap"] = combined_pixmap


    def export_scores(self):
        if not self.need_process_score:
            return
        normal_scores = []
        abnormal_scores = []
        for file_name,data in self.thumbnails.items():
            if self.thumbnails[file_name]["label_state"] =="normal":
                normal_scores.append(self.thumbnails[file_name]["score"])
            elif self.thumbnails[file_name]["label_state"] =="abnormal":
                abnormal_scores.append(self.thumbnails[file_name]["score"])
        
        normal_scores_np = np.array(normal_scores)
        abnormal_scores_np = np.array(abnormal_scores)

        self.labeled_scores.emit(normal_scores_np,abnormal_scores_np)
        

def caculate_heatmap_path(file_path):

    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)
    heatmap_file_name = parts[-2] + parts[-1].replace(".png","_heatmap.png")
    target_dir = os.path.join("cache","heatmap")
    heatmap_path = os.path.join(target_dir,heatmap_file_name)
    return heatmap_path

def caculate_two_mask_path(file_path):
    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)
    heatmap_file_name = parts[-2] + parts[-1].replace(".png","_twomask.png")
    target_dir = os.path.join("cache","mask")
    heatmap_path = os.path.join(target_dir,heatmap_file_name)
    return heatmap_path

def numpy_array_to_qpixmap(numpy_array):
    """将 NumPy 数组转换为 QPixmap"""
    # 1. 确保 NumPy 数组是 uint8 类型并具有 RGB 通道
    if numpy_array.dtype != np.uint8:
        numpy_array = (numpy_array * 255).astype(np.uint8)
    
    # 2. 检查数组的形状，确保是三维 (height, width, channels)
    if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
        height, width, channels = numpy_array.shape
        
        # 3. 将 NumPy 数组转换为 QImage 对象
        qimage = QImage(numpy_array.data, width, height, 3 * width, QImage.Format_RGB888)
        
        # 4. 将 QImage 转换为 QPixmap
        qpixmap = QPixmap.fromImage(qimage)
        
        return qpixmap
    
def make_black_transparent(pixmap):
    # 将 QPixmap 转换为 QImage，以便逐像素处理
    image = pixmap.toImage()
    image = image.convertToFormat(QImage.Format_ARGB32)  # 确保支持透明度

    # 遍历每个像素
    for x in range(image.width()):
        for y in range(image.height()):
            # 获取当前像素的颜色
            color = QColor(image.pixel(x, y))
            
            # 检查是否是黑色像素
            if color.red() == 0 and color.green() == 0 and color.blue() == 0:
                # 设置像素为透明
                image.setPixelColor(x, y, QColor(0, 0, 0, 0))  # 透明颜色

    # 将处理后的 QImage 转换回 QPixmap
    return QPixmap.fromImage(image)
    


# def numpy_array_to_qpixmap_with_transparency(numpy_array):
    
#     img = numpy_array.astype("uint8")
#     qpixmap = QPixmap(QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,QImage.Format_RGB888))
#     # qpixmap = numpy_array_to_qpixmap(numpy_array)



#     return make_black_transparent(qpixmap)

def numpy_array_to_qpixmap_with_transparency(numpy_array):
    # 确保输入为 uint8 类型
    img = numpy_array.astype("uint8")

    # 创建一个新的 ARGB 数组，初始化为全透明
    height, width, _ = img.shape
    argb_array = np.zeros((height, width, 4), dtype=np.uint8)

    # 填充 RGB 通道
    argb_array[..., 0:3] = img[...,::-1]  # RGB

    # 设置 alpha 通道
    # 将黑色像素的 alpha 设置为 0（透明）
    mask = (argb_array[..., 0] == 0) & (argb_array[..., 1] == 0) & (argb_array[..., 2] == 0)
    argb_array[..., 3] = 255  # 默认完全不透明
    argb_array[mask, 3] = 0   # 黑色变为透明

    # 创建 QImage
    qimage = QImage(argb_array.data, width, height, width * 4, QImage.Format_ARGB32)

    # 转换为 QPixmap
    return QPixmap.fromImage(qimage)
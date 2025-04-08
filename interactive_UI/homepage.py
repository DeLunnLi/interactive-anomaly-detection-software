from PyQt5.QtWidgets import QMainWindow, QScrollArea, QVBoxLayout, QHBoxLayout ,QMessageBox,QWidget, QPushButton, QSplitter,QToolBar,QAction,QMenu,QLabel,QSpinBox,QDoubleSpinBox, QComboBox,QDesktopWidget,QGraphicsView,QGraphicsScene
from PyQt5.QtCore import Qt
from .image_utils import Image
from .thumbnail_manage import ThumbnailManager
from .sidebar import statistics_sidebar
from .statistics import Interactive_statistics
from functools import partial
from PyQt5.QtGui import QPalette

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        """
        init parameter

        """
        self.offset_horizontal = 0
        self.offset_vertical = 0
        self.mouse_position = None

        """
        init function class

        """
        # The Image class handles the image display and allows drawing on the image.
        self.image = Image()
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.image)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # The ThumbnailManager class handles the thumbnail part.
        self.thumbnail_manager = ThumbnailManager()
        self.statistics_sidebar = statistics_sidebar()
        self.statistics = Interactive_statistics()
        self.statistics.setFixedWidth(400)
        self.statistics.setVisible(False)
        self.statistics_sidebar.setFixedWidth(20)
        """
        init layout part 

        """
        # set window's name and defaul size
        self.setWindowTitle("Interactive Anomaly Detection Software")
        self.resize(800, 600)
        self.init_menu_bar()
        # set drawing tools bar
        self.drawing_toolbar = self.addToolBar("Drawing Tools")
        self.init_drawing_tool_bar()
        self.drawing_toolbar.hide()

        # set structure of the mainwindow
        central_widget = QWidget()
        button_widget = QWidget()
        main_layout = QHBoxLayout()
        self.splitter = QSplitter(Qt.Horizontal)
        

        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.thumbnail_manager)   
        self.splitter.setStretchFactor(0, 1)  
        self.splitter.setStretchFactor(1, 0)  

        main_layout.addWidget(button_widget)
        main_layout.addWidget(self.splitter)
        main_layout.addWidget(self.statistics)
        main_layout.addWidget(self.statistics_sidebar)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.showMaximized()
        """
        init logic

        """

        # set thumbnail logic
        self.thumbnail_manager.thumbnail_selected.connect(self.image.display_image)
        self.thumbnail_manager.remove_display_image.connect(self.image.remove_display_image)
        self.thumbnail_manager.image_num.connect(self.statistics.update_image_count)
        self.thumbnail_manager.labeled_num.connect(self.statistics.update_labeled_count)
        self.thumbnail_manager.trained_num.connect(self.statistics.update_trained_count)
        self.thumbnail_manager.average_time.connect(self.statistics.update_processing_time)
        self.thumbnail_manager.labeled_scores.connect(self.statistics.plot.update_plot)
        self.thumbnail_manager.matrix.connect(self.statistics.confusion_matrix.update_matrix)


        self.statistics.plot.two_threshold.connect(self.thumbnail_manager.create_mask_heatmap_pixmap)
        # set splitter logic
        self.splitter.splitterMoved.connect(self.thumbnail_manager.adjust_thumbnail_size)
        self.splitter.splitterMoved.connect(self.image.update_displayed_image_size)

        # set signal for information transparent in different part
        self.image.start_annotation_signal.connect(self.show_drawing_ToolBar)
        self.image.end_annotation_signal.connect(self.hide_drawing_ToolBar)
        self.image.save_mask_signal.connect(self.save_mask)
        self.image.focus_zoom_position.connect(self.adjust_scrollbar)
        self.image.focus_zoom_original_position.connect(self.caculate_scrollbar_offset)
        self.image.mask_pixmap_signal.connect(self.thumbnail_manager.add_thumbnail_mask)
        self.image.label_normal_signal.connect(self.thumbnail_manager.update_display_image_label)
        self.image.label_abnormal_signal.connect(self.thumbnail_manager.update_display_image_label)
        self.image.label_default_signal.connect(self.thumbnail_manager.update_display_image_label)

        self.statistics_sidebar.show_statistics_signal.connect(self.show_or_hide_statistics)

    def show_or_hide_statistics(self):
        visible = not self.statistics.isVisible()
        self.statistics.setVisible(visible)


    # Function to adjust the subwindow's size when resize the mainwindow.
    def resizeEvent(self, event):
        # adjust the subwindow's size when resize the mainwindow.
        self.image.update_factor()
        self.thumbnail_manager.adjust_thumbnail_size()

    # Function to set annotation's toolbar
    def init_drawing_tool_bar(self):
        """
        init the toolbar component

        """
        tool_bar_name = QLabel("Annotation Tool Info  ")
        tool_type = QLabel("Tool:")
        self.tool_select_box = QComboBox()
        tools = ["Pen", "Eraser"]
        self.tool_select_box.addItems(tools)
        pen_width = QLabel("Pen Width:")
        pen_width_spinbox = QSpinBox()
        pen_width_spinbox.setRange(5, 100)
        pen_width_spinbox.setSingleStep(5)
        pen_width_spinbox.setValue(10)
        mask_transparency = QLabel("Mask trans:")
        mask_trans_spinbox = QDoubleSpinBox()
        mask_trans_spinbox.setRange(0.0, 100.0)
        mask_trans_spinbox.setSingleStep(5.0)
        mask_trans_spinbox.setValue(50.0)
        mask_trans_spinbox.setSuffix(" %")

        """
        init the toolbar structure

        """
        self.drawing_toolbar.setMinimumHeight(35)
        self.drawing_toolbar.addWidget(tool_bar_name)
        self.drawing_toolbar.addSeparator()
        self.drawing_toolbar.addWidget(tool_type)
        self.drawing_toolbar.addWidget(self.tool_select_box)
        self.drawing_toolbar.addSeparator()
        self.drawing_toolbar.addWidget(pen_width)
        self.drawing_toolbar.addWidget(pen_width_spinbox)
        self.drawing_toolbar.addSeparator()
        self.drawing_toolbar.addWidget(mask_transparency)
        self.drawing_toolbar.addWidget(mask_trans_spinbox)

        """
        init the toolbar logic

        """

        self.tool_select_box.currentIndexChanged.connect(self.on_tool_changed)
        pen_width_spinbox.valueChanged.connect(self.on_pen_width_changed)
        mask_trans_spinbox.valueChanged.connect(self.on_transparency_changed)

    def on_tool_changed(self):
        # 当工具选择改变时发出信号，传递选中的工具名称
        selected_tool = self.tool_select_box.currentText()
        if selected_tool == "Pen":
            self.image.use_pen = True
            self.image.change_pen_mode(Qt.white)
        else:
            self.image.use_pen = False
            self.image.change_pen_mode(Qt.transparent)

    def on_pen_width_changed(self, value):
        # 当笔宽改变时发出信号，传递新的笔宽值
        self.image.change_pen_width(value)
    
    def on_transparency_changed(self, value):
        # 当透明度改变时发出信号，传递新的透明度值
        self.image.change_mask_trans(value)

    def show_drawing_ToolBar(self):
        self.drawing_toolbar.show()
    
    def hide_drawing_ToolBar(self):
        self.drawing_toolbar.hide()

    def save_mask(self):
        self.image.save_image_with_drawing()


    # 存在中央放缩时位置计算的问题，以及划栏位置获取存在一定问题
    def caculate_scrollbar_offset(self,position):
        # 获取当前滚动条位置
        current_horizontal_value = self.scroll_area.horizontalScrollBar().sliderPosition()
        current_vertical_value = self.scroll_area.verticalScrollBar().sliderPosition()

        # 获取显示左上角相对于鼠标之间的偏移量
        self.offset_horizontal = current_horizontal_value - position.x()
        self.offset_vertical = current_vertical_value - position.y()
        self.mouse_position = position

    def adjust_scrollbar(self, position,factor):

        new_position_x = self.mouse_position.x() * factor
        new_position_y = self.mouse_position.y() * factor
        # 更新滚动条位置以保持鼠标位置不变
        self.scroll_area.horizontalScrollBar().setValue(
            max(0, min(self.scroll_area.horizontalScrollBar().maximum(), new_position_x - self.offset_horizontal))
        )
        self.scroll_area.verticalScrollBar().setValue(
            max(0, min(self.scroll_area.verticalScrollBar().maximum(), new_position_y - self.offset_vertical))
    )
        
        # content_height = self.scroll_area.widget().height()
        # visible_height = self.scroll_area.viewport().height()
        # scroll_max_value = self.scroll_area.verticalScrollBar().maximum()
        # scroll_min_value = self.scroll_area.verticalScrollBar().minimum()
        # current_value = self.scroll_area.verticalScrollBar().value()

        # print(f"内容高度: {content_height}, 可见高度: {visible_height}")
        # print(f"滚动条最大值: {scroll_max_value}, 最小值: {scroll_min_value}")
        # print(f"当前滚动条位置: {current_value}")

    def init_menu_bar(self):
        # 创建菜单栏
        menu_bar = self.menuBar()

        # 创建 "workspace" 菜单
        file_menu = menu_bar.addMenu("Workspace")

        # 创建 "Open" 动作
        open_action = QAction("New", self)
        open_action.triggered.connect(self.open_file)  # 绑定动作
        file_menu.addAction(open_action)

        # 创建一个分隔线
        file_menu.addSeparator()

        # 创建 "Save" 动作
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        # 创建 "Close" 动作(修改变量名)
        exit_action = QAction("Close", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 创建 "Close" 动作
        exit_action = QAction("Delete", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 创建一个分隔线
        file_menu.addSeparator()

        exit_action = QAction("Save as New Version", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        exit_action = QAction("Delete Version", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        file_menu.addSeparator()

        exit_action = QAction("Import", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        exit_action = QAction("Export", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        file_menu.addSeparator()

        exit_action = QAction("Quit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # dataset menu
        dataset_menu = menu_bar.addMenu("Dataset")

        # load images action
        load_image_action = QAction("Load Images", self)
        load_image_action.triggered.connect(self.thumbnail_manager.load_images)
        dataset_menu.addAction(load_image_action)

        # remove images action
        remove_selected_images_action = QAction("Remove Selected Images", self)
        remove_selected_images_action.triggered.connect(self.thumbnail_manager.remove_selected_images)
        dataset_menu.addAction(remove_selected_images_action)

        # add separator
        dataset_menu.addSeparator()

        # label as normal image action
        label_selected_image_normal_action = QAction("Label Selected Images--> noraml", self)
        label_selected_image_normal_action.triggered.connect(self.thumbnail_manager.label_selected_image_as_normal)
        dataset_menu.addAction(label_selected_image_normal_action)

        # label as abnormal image action 
        label_selected_image_abnormal_action = QAction("Label Selected Images--> abnoraml", self)
        label_selected_image_abnormal_action.triggered.connect(self.thumbnail_manager.label_selected_image_as_abnormal)
        dataset_menu.addAction(label_selected_image_abnormal_action)

        # set image train state action
        set_images_train_state_action = QAction("Set Selected Images' Train Status", self)
        set_images_train_state_action.triggered.connect(self.redo_action)
        dataset_menu.addAction(set_images_train_state_action)

        # redo_action = QAction("Remove Selected Images' label", self)
        # redo_action.triggered.connect(self.redo_action)
        # edit_menu.addAction(redo_action)

        # redo_action = QAction("Remove Selected Images' Mask", self)
        # redo_action.triggered.connect(self.redo_action)
        # edit_menu.addAction(redo_action)

        # 创建一个分隔线
        dataset_menu.addSeparator()

        redo_action = QAction("Export Model Report", self)
        redo_action.triggered.connect(self.redo_action)
        dataset_menu.addAction(redo_action)

        # 创建 "Edit" 菜单
        view_menu = menu_bar.addMenu("View")

        # 创建 "Undo" 动作
        label_normal_action = QAction("Label as Normal Sample", self)
        label_normal_action.triggered.connect(self.image.trigger_normal_sample)
        view_menu.addAction(label_normal_action)

        label_abnormal_action = QAction("Label as Abnormal Sample", self)
        label_abnormal_action.triggered.connect(self.image.trigger_abnormal_sample)
        view_menu.addAction(label_abnormal_action)

        remove_label_action = QAction("Remove Label", self)
        remove_label_action.triggered.connect(self.image.trigger_default_sample)
        view_menu.addAction(remove_label_action)

        # 创建一个分隔线
        view_menu.addSeparator()

        hide_label_action = QAction("Hide Label", self)
        hide_label_action.triggered.connect(self.image.hide_image_label)
        view_menu.addAction(hide_label_action)

        show_label_action = QAction("Show Label", self)
        show_label_action.triggered.connect(self.image.show_image_label)
        view_menu.addAction(show_label_action)

        view_menu.addSeparator()

        edit_defect_area_action = QAction("Edit Defect Area", self)
        edit_defect_area_action.triggered.connect(self.image.trigger_start_annotation)
        view_menu.addAction(edit_defect_area_action)

        exit_edit_mode_action = QAction("Exit Edit Mode", self)
        exit_edit_mode_action.triggered.connect(self.image.trigger_end_annotation)
        view_menu.addAction(exit_edit_mode_action)

        set_area_as_mask_action = QAction("Set Edited Area as Mask", self)
        set_area_as_mask_action.triggered.connect(self.image.trigger_apply_annotation)
        view_menu.addAction(set_area_as_mask_action)

        remove_mask_action = QAction("Remove Mask", self)
        remove_mask_action.triggered.connect(self.image.trigger_remove_annotation)
        view_menu.addAction(remove_mask_action)

        # 创建一个分隔线
        view_menu.addSeparator()

        undo_action = QAction("Hide Mask", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        view_menu.addAction(undo_action)

        undo_action = QAction("Show Mask", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        view_menu.addAction(undo_action)

        # 创建一个分隔线
        view_menu.addSeparator()

        undo_action = QAction("Image Processing", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        view_menu.addAction(undo_action)

        # 创建 "Edit" 菜单
        tool_menu = menu_bar.addMenu("Controller")

        # 创建 "Undo" 动作
        process_action = QAction("Process", self)
        process_action.triggered.connect(self.thumbnail_manager.process_images)
        tool_menu.addAction(process_action)

        # 创建一个分隔线
        tool_menu.addSeparator()

        # 创建 "Undo" 动作
        train_action = QAction("Train", self)
        train_action.triggered.connect(self.thumbnail_manager.train_model)
        tool_menu.addAction(train_action)

        # 创建 "Undo" 动作
        undo_action = QAction("Terminate", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建 "Undo" 动作
        undo_action = QAction("Reset", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建一个分隔线
        tool_menu.addSeparator()

        # 创建 "Undo" 动作
        undo_action = QAction("Rename Model", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建 "Undo" 动作
        undo_action = QAction("Import Model", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建 "Undo" 动作
        undo_action = QAction("Export Model", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建 "Undo" 动作
        undo_action = QAction("Clone Model", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        tool_menu.addAction(undo_action)

        # 创建 "Edit" 菜单
        help_menu = menu_bar.addMenu("Help")

        # 创建 "Undo" 动作
        undo_action = QAction("Log Information", self)
        undo_action.triggered.connect(self.thumbnail_manager.load_images)
        help_menu.addAction(undo_action)



    def open_file(self):
        QMessageBox.information(self, "Open", "Open file action triggered")

    def save_file(self):
        QMessageBox.information(self, "Save", "Save file action triggered")

    def undo_action(self):
        QMessageBox.information(self, "Undo", "Undo action triggered")

    def redo_action(self):
        QMessageBox.information(self, "Redo", "Redo action triggered")

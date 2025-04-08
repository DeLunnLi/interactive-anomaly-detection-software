import sys

from PyQt5.QtWidgets import QApplication
from interactive_UI.homepage import MainWindow

# version 0.1.0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
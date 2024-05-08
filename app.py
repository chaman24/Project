import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow, QStackedWidget, QSizePolicy, QHBoxLayout, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QDateTime
from prediction import bay_classification
from model_definition import ConvNetWithReLU

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('IIT Hyderabad Parking Real-Time Info')
        self.setGeometry(600, 400, 750, 450)
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        # Background setup using QLabel
        self.backgroundLabel = QLabel(self.centralWidget)
        self.backgroundLabel.setGeometry(0, 0, 750, 450)
        pixmap = QPixmap('tower1.jpg')
        self.backgroundLabel.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding))
        self.backgroundLabel.lower()

        self.layout = QVBoxLayout(self.centralWidget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Left side window for navigation
        self.left_window = QWidget(self.centralWidget)
        self.left_layout = QVBoxLayout(self.left_window)
        self.layout.addWidget(self.left_window)

        # Stacked widget for switching between screens
        self.stacked_widget = QStackedWidget(self.centralWidget)
        self.layout.addWidget(self.stacked_widget)

        self.welcome_screen = WelcomeWindow(self)
        self.stacked_widget.addWidget(self.welcome_screen)

        # Pass self to ParkingInfoWindow to allow access to the main window methods
        self.parking_screen = ParkingInfoWindow(self)
        self.stacked_widget.addWidget(self.parking_screen)

        self.stacked_widget.setCurrentIndex(0)

    def showParkingOptions(self):
        self.parking_options = ParkingOptions(self)
        self.stacked_widget.addWidget(self.parking_options)
        self.stacked_widget.setCurrentWidget(self.parking_options)

    def showParkingScreen(self, block_name):
        # Pass the block name to the ParkingInfoWindow
        self.parking_screen.setBlockName(block_name)
        self.stacked_widget.setCurrentWidget(self.parking_screen)

class WelcomeWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: black;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Add spacer at the top to push buttons to the bottom
        self.layout.addStretch()

        self.info_label = QLabel('Welcome to IIT Hyderabad Parking Real-Time Info', self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.4);
                color: white;
                font-size: 16pt;
            }
        """)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info_label.setFixedHeight(50)
        self.layout.addWidget(self.info_label)

        # Horizontal layout for buttons
        self.button_layout = QHBoxLayout()

        # Add parking area button
        self.parking_area_button = QPushButton('Parking Area', self)
        self.parking_area_button.setStyleSheet("background-color: rgba(255, 255, 255, 0.4); color: black; font-weight: bold; padding: 10px;")
        self.parking_area_button.clicked.connect(self.main_window.showParkingOptions)
        self.parking_area_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.parking_area_button.setCursor(Qt.PointingHandCursor)  # Change cursor to pointing hand
        self.parking_area_button.setObjectName("parkingButton")
        self.button_layout.addWidget(self.parking_area_button)

        # Add exit button
        self.exit_button = QPushButton('Exit', self)
        self.exit_button.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 10px;")
        self.exit_button.clicked.connect(self.main_window.close)
        self.exit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.button_layout.addWidget(self.exit_button)

        self.layout.addLayout(self.button_layout)

        # Add spacer at the bottom to ensure buttons are at the bottom
        self.layout.addStretch()

        # Connect stylesheet for button hover effect
        self.setStyleSheet("""
            QPushButton#parkingButton:hover {
                background-color: rgba(255, 255, 255, 0.7);
            }
        """)

class ParkingOptions(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: black;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.options_list = QListWidget(self)
        self.options_list.setStyleSheet("color: white; font-size: 14pt; background-color: rgba(0, 0, 0, 0.4);")
        self.options_list.addItem("C Block Parking Info")
        self.options_list.addItem("B Block Parking Info")
        self.options_list.itemClicked.connect(self.handleOptionClicked)
        self.layout.addWidget(self.options_list)

    def handleOptionClicked(self, item):
        if item.text() == "C Block Parking Info":
            self.main_window.showParkingScreen("C Block")
        elif item.text() == "B Block Parking Info":
            self.main_window.showParkingScreen("B Block")

class ParkingInfoWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.block_name = ""
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: black;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Adjusted to Expanding
        self.layout.addWidget(self.video_label, alignment=Qt.AlignTop)

        self.result_label = QLabel('Parking Info will be displayed here', self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: white; font-size: 14pt;")
        self.layout.addWidget(self.result_label)

        # Add label for time and date, and version
        self.time_version_label = QLabel(self)
        self.time_version_label.setAlignment(Qt.AlignCenter)
        self.time_version_label.setStyleSheet("color: white; font-size: 12pt;")
        self.layout.addWidget(self.time_version_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.bay_classify = bay_classification()

    def setBlockName(self, block_name):
        self.block_name = block_name

    def showEvent(self, event):
        self.timer.start(30)

    def update_frame(self):
        try:
            frame, occupied_bays, vacant_bays = next(self.bay_classify)
            result_text = f'Occupied Bays: {occupied_bays}, Vacant Bays: {vacant_bays}'
            self.display_frame(frame)
            self.result_label.setText(result_text)
            
            # Update time and version
            current_datetime = QDateTime.currentDateTime()
            version_text = 'Version 1'
            self.time_version_label.setText(f'{current_datetime.toString(Qt.DefaultLocaleLongDate)} | {version_text}')
        except StopIteration:
            self.timer.stop()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytesPerLine = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(700, 350, Qt.KeepAspectRatioByExpanding)
        if p.height() > 350:
            crop_pixels = (p.height() - 350) // 2
            p = p.copy(0, crop_pixels, 700, 350)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.timer.stop()

    def goBack(self):
        self.timer.stop()
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.welcome_screen)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

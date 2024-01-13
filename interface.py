import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Main Interface")
        self.setGeometry(0, 0, 600, 400)
        self.center_on_screen()

        # Set up main layout
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # Add title, subtitle, background image, and menu buttons
        title_label = QLabel("Main Title", self)
        subtitle_label = QLabel("Subtitle", self)
        background_image = QLabel(self)
        # background_image.setPixmap(QPixmap("images/hand.jpeg").scaledToWidth(self.width()).scaledToHeight(self.height()))

        menu_button1 = QPushButton("Select Images", self)
        menu_button2 = QPushButton("Compare Images", self)
        menu_button3 = QPushButton("Show Result", self)

        # creating a label widget for background 
        # self.label_2 = QLabel(self)
        # self.label_2.setGeometry(0,0,600,400) 
        # moving position 
        # self.label_2.move(160, 170)
        # setting up the border and adding image to background 
        # self.label_2.setStyleSheet("background-image : url(images/mark.jpeg); border : 2px solid blue") 
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        # main_layout.addWidget(background_image)
        main_layout.addWidget(menu_button1)
        main_layout.addWidget(menu_button2)
        main_layout.addWidget(menu_button3)

        # Set up connections
        menu_button1.clicked.connect(self.show_image_selection)

        self.setCentralWidget(main_widget)

    def show_image_selection(self):
        self.image_selection = ImageSelectionWindow(self)
        self.image_selection.show()
        
        
    def center_on_screen(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        x = int((screen_geometry.width() - self.width()) / 2)
        y = int((screen_geometry.height() - self.height()) / 2)
        self.move(x, y)

class ImageSelectionWindow(QWidget):
    def __init__(self, parent=None):
        super(ImageSelectionWindow, self).__init__(parent)

        self.setWindowTitle("Image Selection")
        self.setGeometry(0, 0, 600, 400)

        # Set up layout
        layout = QVBoxLayout(self)

        select_button = QPushButton("Select Images", self)
        select_button.clicked.connect(self.select_images)

        layout.addWidget(select_button)

    def select_images(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            print(f"Selected Images: {selected_files}")
            
# class ImageSelectionWindow(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.initUI()

#     def initUI(self):
#         # Create layouts
#         main_layout = QVBoxLayout()
#         box_layout1 = QVBoxLayout()
#         box_layout2 = QVBoxLayout()

#         # Create image labels and buttons for Box 1
#         label1 = QLabel('Image 1')
#         self.image_label1 = QLabel()
#         self.load_button1 = QPushButton('Click here to select image')
#         self.load_button1.clicked.connect(self.loadImage1)

#         # Create image labels and buttons for Box 2
#         label2 = QLabel('Image 2')
#         self.image_label2 = QLabel()
#         self.load_button2 = QPushButton('Click here to select image')
#         self.load_button2.clicked.connect(self.loadImage2)

#         # Add widgets to layouts
#         box_layout1.addWidget(label1)
#         box_layout1.addWidget(self.image_label1)
#         box_layout1.addWidget(self.load_button1)

#         box_layout2.addWidget(label2)
#         box_layout2.addWidget(self.image_label2)
#         box_layout2.addWidget(self.load_button2)

#         # Add box layouts to main layout
#         main_layout.addLayout(box_layout1)
#         main_layout.addLayout(box_layout2)

#         self.setLayout(main_layout)
#         self.setWindowTitle('Image Selection Window')

#     def loadImage1(self):
#         file_dialog = QFileDialog()
#         image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp *.gif)')
#         if image_path:
#             pixmap = QPixmap(image_path)
#             self.image_label1.setPixmap(pixmap.scaled(200, 200))  # Resize for display

#     def loadImage2(self):
#         file_dialog = QFileDialog()
#         image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp *.gif)')
#         if image_path:
#             pixmap = QPixmap(image_path)
#             self.image_label2.setPixmap(pixmap.scaled(200, 200))  # Resize for display

class ResultWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ResultWindow, self).__init__(parent)

        self.setWindowTitle("Result Interface")
        self.setGeometry(300, 300, 600, 400)

        # Set up result layout
        result_label = QLabel("Result Image", self)
        result_image = QLabel(self)
        result_image.setPixmap(QPixmap("result_image.jpg").scaledToWidth(600))

        self.setCentralWidget(result_label)
        self.setCentralWidget(result_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

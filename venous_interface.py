import sys
import qrc_resources
from PyQt5.QtGui import QIcon
from functools import partial
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5 import QtCore, QtGui, QtWidgets

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application d'opérations sur les empreintes veineuses")
        self.resize(500, 700)
        self.center_on_screen()
        label = QLabel(self)
        pixmap = QPixmap("/home/mick/Documents/Aides/Placide/VenousMark/images/hand_3.jpeg")
        # pixmap = pixmap.scaled(500, 700)
        pixmap = pixmap.scaledToWidth(self.width())
        label.setPixmap(pixmap)
        self.setCentralWidget(label)
        # self.resize(pixmap.width(), pixmap.height())
        centralWidget = QLabel("Hello, World")
        centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(centralWidget)
        text_label = QLabel("Vein Recogniton System", self)
        text_label.setFont(QFont("Georgia", 32, weight = 50, italic = True))  # Adjust font size and style as needed
        text_label.setStyleSheet("color: black")  # Set text color
        text_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(text_label)
        text_label.move(8, 60)
        text_label.adjustSize()
        sub_text_label = QLabel("Une approche basée sur les minuties", self)
        sub_text_label.setFont(QFont("Georgia", 11, weight = 50, italic = True))  # Adjust font size and style as needed
        sub_text_label.setStyleSheet("color: black")  # Set text color
        sub_text_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        sub_text_label.move(230, 118)
        sub_text_label.adjustSize()
        self._createActions()
        self._createMenuBar()
        self._connectActions()
        self._createStatusBar()
        
    
    def _createActions(self):
        self.newAction = QAction("&New", self)
        new_action_tip = "Create a new file"
        self.newAction.setStatusTip(new_action_tip)
        self.newAction.setToolTip(new_action_tip)
        
        self.openAction = QAction("&Open", self)
        openAction_tip = "Open action tip."
        self.openAction.setStatusTip(openAction_tip)
        self.openAction.setToolTip(openAction_tip)
        
        self.load_image = QAction(QIcon(":upload.svg"), "&Charger une image", self)
        load_image_tip = "Charger une image depuis le stockage local."
        self.load_image.setStatusTip(load_image_tip)
        self.load_image.setToolTip(load_image_tip)
        
        self.minutia = QAction(QIcon(":features.svg"), "&Extraction de minuties", self)
        minutia_tip = "Minutia tip."
        self.minutia.setStatusTip(minutia_tip)
        self.minutia.setToolTip(minutia_tip)
        
        self.comparison = QAction(QIcon(":comparison-3.svg"), "&Comparaison", self)
        comparison_tip = "Comparison tip."
        self.comparison.setStatusTip(comparison_tip)
        self.comparison.setToolTip(comparison_tip)
        
        self.roc = QAction(QIcon(":curve.svg"), "&Courbe ROC", self)
        roc_tip = "Tracer la courbe ROC"
        self.roc.setStatusTip(roc_tip)
        self.roc.setToolTip(roc_tip)
        
        self.exit_action = QAction("&Fermer", self)
        exit_tip = "Fermer l'application."
        self.exit_action.setStatusTip(exit_tip)
        self.exit_action.setToolTip(exit_tip)
        
        self.help_action = QAction(QIcon(":about-1.svg"), "&Aide", self)
        help_tip = "Accéder à l'aide."
        self.help_action.setStatusTip(help_tip)
        self.help_action.setToolTip(help_tip)
        
        self.about_action = QAction(QIcon(":about.svg"), "&A propos", self)
        about_tip = "A propos de l'application."
        self.about_action.setStatusTip(about_tip)
        self.about_action.setToolTip(about_tip)
    
    def _connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.showDialog)
        self.load_image.triggered.connect(self.LoadImage)
        self.minutia.triggered.connect(self.Minutia)
        self.exit_action.triggered.connect(self.close)
        self.comparison.triggered.connect(self.Comparison)
        self.roc.triggered.connect(self.ROCCurve)
        self.help_action.triggered.connect(self.helpContent)
        self.about_action.triggered.connect(self.about)
    
    def showDialog(self):
        dialog = QDialog()
        b1 = QPushButton("ok", dialog)
        b1.move(50,50)
        dialog.setWindowTitle("Dialog")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.exec_()
    
    def LoadImage(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            print(f"Selected Images: {selected_files}")
    
    def newFile(self):
        self.centralWidget.setText("<b>File > New</b> clicked")
        
    def openFile(self):
        self.centralWidget.setText("<b>File > Open...</b> clicked")
    
    def Minutia(self):
        # self.centralWidget.setText("<b>Operations > Minutia</b> clicked")
        self.statusbar = self.statusBar()
        self.minutia_window = MinutiaWindow(self)
        self.minutia_window.show()
    
    def Comparison(self):
        self.statusbar = self.statusBar()
        self.comparison_window = ComparisonWindow(self)
        self.comparison_window.show()
    
    def ROCCurve(self):
        self.centralWidget.setText("<b>Operations > ROC</b> clicked")
    
    def helpContent(self):
        self.centralWidget.setText("<b>Help > Help Content...</b> clicked")

    def about(self):
        self.centralWidget.setText("<b>Help > About...</b> clicked")
    
    def _createMenuBar(self):
        menuBar = self.menuBar()
        file_menu = QMenu("&Accueil", self)
        menuBar.addMenu(file_menu)
        file_menu.addAction(self.newAction)
        file_menu.addAction(self.openAction)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        operations_menu = menuBar.addMenu("&Opérations")
        operations_menu.addAction(self.load_image)
        operations_menu.addAction(self.minutia)
        operations_menu.addAction(self.comparison)
        operations_menu.addAction(self.roc)
        help_menu = menuBar.addMenu("&Aide")
        help_menu.addAction(self.help_action)
        # about_menu = menuBar.addMenu("&À propos")
        help_menu.addAction(self.about_action)
        
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready", 10000)
        
    def center_on_screen(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        x = int((screen_geometry.width() - self.width()) / 2)
        y = int((screen_geometry.height() - self.height()) / 2)
        self.move(x, y)
        

class MinutiaWindow(QMainWindow):
    def __init__(self, parent=Window):
        super(MinutiaWindow, self).__init__(parent)
        self.setWindowTitle("Interface d'extraction des Minuties")
        self.resize(880, 450)
        self.center_on_screen()
        # self.centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(self.centralWidget)
        self.select_image_label = QLabel(self)
        self.select_image_label.resize(400, 350)
        self.select_image_label.move(20,20)
        self.minutia_image_label = QLabel(self)
        self.minutia_image_label.resize(400, 350)
        self.minutia_image_label.move(500,20)
        # self.setCentralWidget(label)
        select_button = QPushButton("Sélectionner une Image", self)
        select_button.clicked.connect(self.select_images)
        select_button.move(115, 400)
        select_button.adjustSize()
        extraction_button = QPushButton("Extraire les Minuties", self)
        extraction_button.move(590, 400)
        extraction_button.adjustSize()
        
        self._createStatusBar()

    def select_images(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec_():
            selected_image = file_dialog.selectedFiles()
            print(f"Selected Images: {selected_image}")
            # label = QLabel(self)
            pixmap = QPixmap(selected_image[0])
            smaller_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.select_image_label.setPixmap(smaller_pixmap)
            self.minutia_image_label.setPixmap(smaller_pixmap)
            # label.resize(480, 480)
            # self.setCentralWidget(self.select_image_label)
            
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Sélectionner une image et Extraire les minuties.", 10000)
           
    def center_on_screen(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        x = int((screen_geometry.width() - self.width()) / 2)
        y = int((screen_geometry.height() - self.height()) / 2)
        self.move(x, y)


class ComparisonWindow(QMainWindow):
    def __init__(self, parent=Window):
        super(ComparisonWindow, self).__init__(parent)
        
        self.setWindowTitle("Interface de Comparaison")
        self.resize(880, 400)
        self.center_on_screen()
        # self.centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(self.centralWidget)
        self.select_image_label_1 = QLabel(self)
        self.select_image_label_1.resize(400, 350)
        self.select_image_label_1.move(20,40)
        self.select_image_label_2 = QLabel(self)
        self.select_image_label_2.resize(400, 350)
        self.select_image_label_2.move(500,40)
        # self.setCentralWidget(label)
        select_button_1 = QPushButton("Sélectionner une Image", self)
        select_button_1.clicked.connect(self.select_image_1)
        select_button_1.move(115, 10)
        select_button_1.adjustSize()
        select_button_2 = QPushButton("Sélectionner une Image", self)
        select_button_2.clicked.connect(self.select_image_2)
        select_button_2.move(590, 10)
        select_button_2.adjustSize()
        comparison_button = QPushButton("Comparer", self)
        comparison_button.clicked.connect(self.comparison)
        comparison_button.move(395, 150)
        comparison_button.adjustSize()
        
        
        self.text_label = QLabel("...", self)
        self.text_label.setFont(QFont("Georgia", 18, weight = 50, italic = True))  # Adjust font size and style as needed
        self.text_label.setStyleSheet("color: green")  # Set text color
        self.text_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(text_label)
        self.text_label.move(410, 180)
        self.text_label.adjustSize()
        
        self._createStatusBar()
        
    def comparison(self):
        self.text_label.setText("20 %")
        self.text_label.adjustSize()
        
    def select_image_1(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            selected_image = file_dialog.selectedFiles()
            print(f"Selected Image 1: {selected_image[0]}")
            pixmap = QPixmap(selected_image[0])
            smaller_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.select_image_label_1.setPixmap(smaller_pixmap)
            
    def select_image_2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            selected_image = file_dialog.selectedFiles()
            print(f"Selected Image 2: {selected_image[0]}")
            pixmap = QPixmap(selected_image[0])
            smaller_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.select_image_label_2.setPixmap(smaller_pixmap)
            
            
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Sélectionner une image et Extraire les minuties.", 10000)
           
    def center_on_screen(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        x = int((screen_geometry.width() - self.width()) / 2)
        y = int((screen_geometry.height() - self.height()) / 2)
        self.move(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
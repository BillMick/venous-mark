import sys
import qrc_resources
from PyQt5.QtGui import QIcon
from functools import partial
from PyQt5.QtWidgets import QToolBar
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtWidgets import QMenuBar
from PyQt5.QtWidgets import QMenu
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("Python Menus & Toolbars")
        self.resize(800, 500)
        self.center_on_screen()
        self.centralWidget = QLabel("Hello, World")
        self.centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setCentralWidget(self.centralWidget)
        self._createActions()
        self._createToolBars()
        self._createMenuBar()
        # self._createContextMenu()
        self._connectActions()
        self._createStatusBar()
    
    def showDialog(self):
        dialog = QDialog()
        b1 = QPushButton("ok", dialog)
        b1.move(50,50)
        dialog.setWindowTitle("Dialog")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.exec_()
    
    def center_on_screen(self):
        screen_geometry = QDesktopWidget().availableGeometry()
        x = int((screen_geometry.width() - self.width()) / 2)
        y = int((screen_geometry.height() - self.height()) / 2)
        self.move(x, y)
        
    def getWordCount(self):
        # Logic for computing the word count goes here...
        return 42
    
    # 1. Using .statusBar()
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        # Adding a temporary message
        self.statusbar.showMessage("Ready", 10000)
        # Adding a permanent message
        self.wcLabel = QLabel(f"{self.getWordCount()} Words")
        self.statusbar.addPermanentWidget(self.wcLabel)
        
    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        # Adding an Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        # Adding a separator
        fileMenu.addSeparator()
        fileMenu.addAction(self.fileAction)
        fileMenu.addAction(self.visualizeAction)
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.downloadAction)
        editMenu.addAction(self.uploadAction)
        # Adding a separator
        editMenu.addSeparator()
        editMenu.addAction(self.pictureAction)
        # Help menu
        helpMenu = menuBar.addMenu(QIcon(":upload.svg"), "&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)
        # Find and Replace submenu in the Edit menu
        findMenu = editMenu.addMenu("Find and Replace")
        findMenu.addAction("Find...")
        findMenu.addAction("Replace...")
    
    def populateOpenRecent(self):
        # Step 1. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 2. Dynamically create the actions
        actions = []
        filenames = [f"File-{n}" for n in range(5)]
        for filename in filenames:
            action = QAction(filename, self)
            action.triggered.connect(partial(self.openRecentFile, filename))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)
        
    def _createToolBars(self):
        # Using a title
        fileToolBar = self.addToolBar("File")
        fileToolBar.setMovable(False)
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.fileAction)
        fileToolBar.addAction(self.visualizeAction)
        # Using a QToolBar object
        editToolBar = QToolBar("Edit", self)
        self.addToolBar(editToolBar)
        editToolBar.addAction(self.downloadAction)
        editToolBar.addAction(self.uploadAction)
        editToolBar.addAction(self.pictureAction)
        # Using a QToolBar object and a toolbar area
        helpToolBar = QToolBar("Help", self)
        self.addToolBar(Qt.LeftToolBarArea, helpToolBar)
        # Adding a widget to the Edit toolbar
        self.fontSizeSpinBox = QSpinBox()
        self.fontSizeSpinBox.setFocusPolicy(Qt.NoFocus)
        editToolBar.addWidget(self.fontSizeSpinBox)
        
    def _createActions(self):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        # Adding help tips
        newTip = "Create a new file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)
        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.fileAction = QAction(QIcon(":file-new.svg"), "&Open...", self)
        self.visualizeAction = QAction(QIcon(":visualization.svg"), "&Visualize", self)
        self.downloadAction = QAction(QIcon(":download.svg"), "&Download", self)
        self.uploadAction = QAction(QIcon(":upload.svg"), "&Upload", self)
        self.pictureAction = QAction(QIcon(":picture.svg"), "&Picture", self)
        self.helpContentAction = QAction("&Help Content", self)
        self.exitAction = QAction("&Exit", self)
        self.aboutAction = QAction("&About", self)

    def _createContextMenu(self):
        # Setting contextMenuPolicy
        self.centralWidget.setContextMenuPolicy(Qt.ActionsContextMenu)
        # Populating the widget with actions
        self.centralWidget.addAction(self.newAction)
        self.centralWidget.addAction(self.openAction)
        self.centralWidget.addAction(self.fileAction)
        self.centralWidget.addAction(self.visualizeAction)
        self.centralWidget.addAction(self.downloadAction)
        self.centralWidget.addAction(self.uploadAction)
    
    def _connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.showDialog)
        self.fileAction.triggered.connect(self.FileAction)
        self.exitAction.triggered.connect(self.close)
        # Connect Edit actions
        self.visualizeAction.triggered.connect(self.VisualizeAction)
        self.downloadAction.triggered.connect(self.DownloadAction)
        self.uploadAction.triggered.connect(self.UploadAction)
        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
    
    def openRecentFile(self, filename):
        # Logic for opening a recent file goes here...
        self.centralWidget.setText(f"<b>{filename}</b> opened")
        
    def newFile(self):
        # Logic for creating a new file goes here...
        self.centralWidget.setText("<b>File > New</b> clicked")

    def openFile(self):
        # Logic for opening an existing file goes here...
        self.centralWidget.setText("<b>File > Open...</b> clicked")

    def FileAction(self):
        # Logic for saving a file goes here...
        self.centralWidget.setText("<b>File > File</b> clicked")

    def VisualizeAction(self):
        # Logic for copying content goes here...
        self.centralWidget.setText("<b>Edit > Visualization</b> clicked")

    def DownloadAction(self):
        # Logic for pasting content goes here...
        self.centralWidget.setText("<b>Edit > Download</b> clicked")

    def UploadAction(self):
        # Logic for cutting content goes here...
        self.centralWidget.setText("<b>Edit > Upload</b> clicked")

    def helpContent(self):
        # Logic for launching help goes here...
        self.centralWidget.setText("<b>Help > Help Content...</b> clicked")

    def about(self):
        # Logic for showing an about dialog content goes here...
        self.centralWidget.setText("<b>Help > About...</b> clicked")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
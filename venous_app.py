import sys
import qrc_resources
from functools import partial
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from pylab import gray, imshow, show
from matplotlib.widgets import Slider, Button
from skimage.io import imread, imshow
from PIL import Image, ImageChops
from skimage import data, io, color, feature, filters, morphology
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
import cv2
import numpy as np
from itertools import product
from scipy.stats import describe
import skimage.morphology
import math

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            (blkRows, blkCols) = np.shape(block)
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 3):
                angle.append(float('nan'))
            return (angle)

    def __getTerminationBifurcation(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1

        self._mask = skimage.morphology.convex_hull_image(self._mask > 0)
        self._mask = skimage.morphology.erosion(self._mask, skimage.morphology.square(5))
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                D[i][j] = dist
                if(dist < self._spuriousMinutiaeThresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid'])
                img[X,Y] = 1

        img = np.uint8(img)
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if(len(angle) == 3):
                FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
        return (FeaturesTerm, FeaturesBif)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcation()

        FeaturesTerm, FeaturesBif = self.__performFeatureExtraction()
        return (FeaturesTerm, FeaturesBif)

    def showResults(self, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        disp_img = np.zeros((rows, cols, 3), np.uint8)
        disp_img[:, :, 0] = 255 * self._skel
        disp_img[:, :, 1] = 255 * self._skel
        disp_img[:, :, 2] = 255 * self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(disp_img, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(disp_img, (rr, cc), (255, 0, 0))

        cv2.imshow('output', disp_img)
        cv2.waitKey(0)

    def saveResult(self, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        disp_img = np.zeros((rows, cols, 3), np.uint8)
        disp_img[:, :, 0] = 255 * self._skel
        disp_img[:, :, 1] = 255 * self._skel
        disp_img[:, :, 2] = 255 * self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(disp_img, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(disp_img, (rr, cc), (255, 0, 0))
        output_image_path = 'result.png'
        cv2.imwrite(output_image_path, disp_img)
        return output_image_path

def extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img;
    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)
    if (saveResult):
        output_image_path = feature_extractor.saveResult(FeaturesTerm, FeaturesBif)
    if (showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif)
    return output_image_path, (FeaturesTerm, FeaturesBif)

def extract_minutiaecido(image_path):
    img = cv2.imread(image_path)
    # Convertir l'image en niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Réduction du bruit sur l'image en niveau de gris (debruitage par moyenne non locale)
    noiseReduced = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Egalisation d'histogramme avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    finalImg = clahe.apply(noiseReduced)
    # Filtrage de Gabor
    ksize, psi, gamma, sigma, lambd = 35, 0, .5, 5, 15
    theta_list = np.deg2rad([18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
    K = len(theta_list)
    filters = []
    for theta in theta_list:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)

    filtered_images = []
    for kernel in filters:
        filtered = cv2.filter2D(finalImg, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered)

    fused_image = np.minimum.reduce(filtered_images)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
    # Sommation de l'image initiale et de l'image filtrée
    img_weighted = cv2.addWeighted(img, 0.3, fused_image, 0.7, 0)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2GRAY)
    # Segmentation par seuillage adaptatif
    thresholded = cv2.adaptiveThreshold(img_weighted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 4)
    # Réduction du bruit après segmentation
    noiseReduced = cv2.fastNlMeansDenoising(thresholded, None, h=20, templateWindowSize=5, searchWindowSize=20)
    # Egalisation d'histogramme après seuillage
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(9, 9))
    finalImg = clahe.apply(noiseReduced)
    # Définir le noyau pour l'érosion
    kernel = np.ones((5, 5), np.uint8)
    # Appliquer l'érosion à l'image
    erosion = cv2.erode(finalImg, kernel, iterations=1)
    # Appliquer l'algorithme de squelettisation de Zhang-Suen
    skeleton = cv2.ximgproc.thinning(erosion, cv2.ximgproc.THINNING_ZHANGSUEN)
    # Afficher l'image avec les minuties
    output_image_path, (FeaturesTerm, FeaturesBif) = extract_minutiae_features(
        skeleton,  # Charger l'image
        spuriousMinutiaeThresh=10,  # Valeur seuil pour les minuties superflues
        invertImage=False,  # Inverser ou non les valeurs des pixels de l'image
        showResult=False,  # Afficher ou non l'image avec les minuties
        saveResult=True  # Enregistrer ou non l'image avec les minuties
    )
    return output_image_path

def extract_minutiaepathimage1(img_path1):
    img1 = cv2.imread(img_path1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    noiseReduced1 = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    finalImg1 = clahe.apply(noiseReduced1)
    ksize, psi, gamma, sigma, lambd = 35, 0, .5, 5, 15
    theta_list = np.deg2rad([18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
    K = len(theta_list)
    filters = []
    for theta in theta_list:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
    filtered_images = []
    for kernel in filters:
        filtered = cv2.filter2D(finalImg1, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered)
    fused_image = np.minimum.reduce(filtered_images)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
    img_weighted = cv2.addWeighted(img1, 0.3, fused_image, 0.7, 0)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(img_weighted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 4)
    noiseReduced = cv2.fastNlMeansDenoising(thresholded, None, h=20, templateWindowSize=5, searchWindowSize=20)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(9, 9))
    finalImg = clahe.apply(noiseReduced)
    skeleton = cv2.ximgproc.thinning(finalImg, cv2.ximgproc.THINNING_ZHANGSUEN)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_weighted, mask=None)
    print("Nombre de minuties detectées (image 1)", len(kp1))
    print("Matrice de description (image 1)", des1)
    return kp1, des1

def extract_minutiaepathimage2(img_path2):
    img2 = cv2.imread(img_path2)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    noiseReduced1 = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    finalImg1 = clahe.apply(noiseReduced1)
    ksize, psi, gamma, sigma, lambd = 35, 0, .5, 5, 15
    theta_list = np.deg2rad([18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
    K = len(theta_list)
    filters = []
    for theta in theta_list:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
    filtered_images = []
    for kernel in filters:
        filtered = cv2.filter2D(finalImg1, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered)
    fused_image = np.minimum.reduce(filtered_images)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
    img_weighted = cv2.addWeighted(img2, 0.3, fused_image, 0.7, 0)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(img_weighted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 4)
    noiseReduced = cv2.fastNlMeansDenoising(thresholded, None, h=20, templateWindowSize=5, searchWindowSize=20)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(9, 9))
    finalImg = clahe.apply(noiseReduced)
    skeleton = cv2.ximgproc.thinning(finalImg, cv2.ximgproc.THINNING_ZHANGSUEN)
    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img_weighted, mask=None)
    print("Nombre de minuties detectées (image 2):", len(kp2))
    print("Matrice de description (image 2)", des2)
    return kp2, des2

def extract_minutiae(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noiseReduced1 = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    finalImg1 = clahe.apply(noiseReduced1)
    ksize = 35
    psi = 0
    gamma = 0.5
    sigma = 5
    theta_list = np.deg2rad([18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
    lambd = 15
    K = len(theta_list)
    filters = []
    for theta in theta_list:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
    filtered_images = []
    for kernel in filters:
        filtered = cv2.filter2D(finalImg1, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered)
    fused_image = np.minimum.reduce(filtered_images)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
    img_weighted = cv2.addWeighted(img, 0.3, fused_image, 0.7, 0)
    img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(img_weighted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 4)
    noiseReduced = cv2.fastNlMeansDenoising(thresholded, None, h=20, templateWindowSize=5, searchWindowSize=20)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(9, 9))
    finalImg = clahe.apply(noiseReduced)
    skeleton = cv2.ximgproc.thinning(finalImg, cv2.ximgproc.THINNING_ZHANGSUEN)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_weighted, mask=None)
    return des1

def match_descriptors(des1, des2):
    nb_points_interet = min(len(des1), len(des2))
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.88 * n.distance:
            good_matches.append(m)
    taux_correspondance = len(good_matches) / nb_points_interet
    taux_correspondance = taux_correspondance * 100
    taux_correspondance = int(taux_correspondance)
    return taux_correspondance

def match_images(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    des1 = extract_minutiae(img1)
    des2 = extract_minutiae(img2)
    taux_correspondance = match_descriptors(des1, des2)
    return taux_correspondance

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application d'opérations sur les empreintes veineuses")
        self.resize(500, 700)
        self.center_on_screen()
        label = QLabel(self)
        pixmap = QPixmap("images/les_veines_de_la_main.jpg")
        # pixmap = pixmap.scaled(500, 700)
        pixmap = pixmap.scaledToWidth(self.width())
        label.setPixmap(pixmap)
        self.setCentralWidget(label)
        # self.resize(pixmap.width(), pixmap.height())
        centralWidget = QLabel("Hello, World")
        centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(centralWidget)
        text_label = QLabel("Vein Recogniton System", self)
        text_label.setFont(QFont("Georgia", 32, weight = 50, italic = True))
        text_label.setStyleSheet("color: black")
        text_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.setCentralWidget(text_label)
        text_label.move(8, 60)
        text_label.adjustSize()
        sub_text_label = QLabel("A minutiae-based approach", self)
        sub_text_label.setFont(QFont("Georgia", 11, weight = 50, italic = True))
        sub_text_label.setStyleSheet("color: black")
        sub_text_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        sub_text_label.move(300, 118)
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
        self.chemin_image = " "
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
        extraction_button.clicked.connect(self.select_images_test)
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
            self.chemin_image = selected_image[0]
            #####################################################################""""
            # minutia_image = extract_minutiae(selected_image[0])
            # minutia_pixmap = QPixmap(minutia_image)
            # smaller_minutia_pixmap = minutia_pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            # self.minutia_image_label.setPixmap(smaller_minutia_pixmap)
            #########################################################################
            # label.resize(480, 480)
            # self.setCentralWidget(self.select_image_label)
            
    def select_images_test(self):
        output_image = extract_minutiaecido(self.chemin_image)
        minutia_pixmap = QPixmap(output_image)
        smaller_minutia_pixmap = minutia_pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.minutia_image_label.setPixmap(smaller_minutia_pixmap)
        #########################################################################
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
        self.image_path_1 = ""
        self.image_path_2 = ""
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
        if self.image_path_1 and self.image_path_2:
            # Effectuer la comparaison des images
            resultat_comparaison = match_images(self.image_path_1, self.image_path_2)
            print("Taux de correspondance des minuties entre les deux images:", resultat_comparaison)

            # Afficher le résultat de la comparaison dans l'interface utilisateur
            self.text_label.setText(f"{resultat_comparaison:}%")
            self.text_label.adjustSize()


        else:
            print("Veuillez sélectionner deux images pour la comparaison.")
    
    def select_image_1(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            selected_image = file_dialog.selectedFiles()
            print(f"Selected Image 1: {selected_image[0]}")
            self.image_path_1 = selected_image[0]
            pixmap = QPixmap(selected_image[0])
            smaller_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.select_image_label_1.setPixmap(smaller_pixmap)
            kp1, des1 = extract_minutiaepathimage1(self.image_path_1)

    def select_image_2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            selected_image = file_dialog.selectedFiles()
            print(f"Selected Image 2: {selected_image[0]}")
            self.image_path_2 = selected_image[0]
            pixmap = QPixmap(selected_image[0])
            smaller_pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.select_image_label_2.setPixmap(smaller_pixmap)
            kp2, des2 = extract_minutiaepathimage2(self.image_path_2)
            
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
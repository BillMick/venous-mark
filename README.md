# venous-mark
Venous Mark app made with python and pyQt.

## Description
_Vein biometrics is an innovative technology that exploits an individual's unique vein network to recognize and authenticate him or her. Unlike other biometric techniques such as fingerprinting or facial recognition, vein biometrics is based on an internal characteristic that is invisible to the naked eye, making it particularly secure and difficult to falsify._

## Image processing
Application of image processing basic principles:
- to define region of interest
- to apply preprocessing (gabor filtering, binarization, etc.)
- to extract minutiae
- to compare two vein footprints
- etc.

## Structure
- __venous_app.py__ is the main module to execute.
- The folder __Test images__ contains images to try the app. In this folder, there are four images (two belong to _person A_).

## Work well
- Minutiae extraction
- Comparison of two vein footprints
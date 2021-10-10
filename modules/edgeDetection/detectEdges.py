'''
Created on 20.01.2020

@author: grustentier 
 
'''

print(""" 


  ___      _                      ___          _                _                 
 | __|  __| |  __ _   ___   ___  |   \   ___  | |_   ___   __  | |_   ___   _ _   
 | _|  / _` | / _` | / -_) |___| | |) | / -_) |  _| / -_) / _| |  _| / _ \ | '_|  
 |___| \__,_| \__, | \___|       |___/  \___|  \__| \___| \__|  \__| \___/ |_|    
              |___/                                                               
  _                ___                     _                  _     _             
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _ 
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|  
         |__/                                                                     


""")
import os 
import cv2
import numpy
import shutil
import argparse
 
parser = argparse.ArgumentParser(description='Code for edge detection. by Grustentier')
parser.add_argument('--input_dir', default='', type=str, help='Path to input_dir dir with images to correct gamma.')
parser.add_argument('--export_dir', default='', type=str, help='The export/export_dir directory')
parser.add_argument('--method', default='canny', type=str, help='Edge detection methods to choose: canny, sobel and laplacian')
parser.add_argument('--sobel_kernel', default=5, type=int, help='The sobel kernel size')
parser.add_argument('--canny_x', default=120, type=int, help='The x gradient for canny algorithm')
parser.add_argument('--canny_y', default=120, type=int, help='The y gradient for canny algorithm')
parser.add_argument('--canny_smoothing', default=False, type=str, help='True for applying bilateral filter at first')
parser.add_argument('--invert', default=False, type=str, help='True for inverting image')
arguments = parser.parse_args()


FILE_TYPES = ["bmp","jpg","jpeg","png"]

def boolean_string(s):
    print(str(s).lower())
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'

def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.input_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths

def copyTree():
    if os.path.exists(arguments.export_dir):
        try:
            shutil.rmtree(arguments.export_dir)
        except OSError as e:
            print("Error: %s : %s" % (arguments.export_dir, e.strerror))   
 
    shutil.copytree(arguments.input_dir,arguments.export_dir,ignore=ignore_files)
    
def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%") 

def processSobel(img):
    """create edged image by sobel method"""
    sobelImage = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=arguments.sobel_kernel)
    sobelImageX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=arguments.sobel_kernel)
    sobelImageY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=arguments.sobel_kernel)   
    
    if boolean_string(arguments.invert) is True:
        """create inverted edged image by sobel method"""
        sobelImage = numpy.invert(sobelImage)
        sobelImageX = numpy.invert(sobelImageX)
        sobelImageY = numpy.invert(sobelImageY)
        
    return sobelImage, sobelImageX, sobelImageY 

def processLaplacian(img):
    """create edged image by laplacian method"""
    laplacianImage = cv2.Laplacian(img, cv2.CV_64F) 

    if boolean_string(arguments.invert) is True:
        """create inverted edged image by laplacian method"""
        #laplacianImage = cv2.cvtColor(laplacianImage, cv2.COLOR_BGR2RGB)
        laplacianImage = numpy.invert(laplacianImage.astype('uint8') * 255)
        
    return laplacianImage

def processCanny(img):  
    """create edged image by canny method"""
    if boolean_string(arguments.canny_smoothing) is True:
        # Smoothing without removing edges.
        img = cv2.bilateralFilter(img, 7, 50, 50)
    
    cannyImage = cv2.Canny(img,arguments.canny_x,arguments.canny_y)   
    
    if boolean_string(arguments.invert) is True:
        """create inverted edged image by canny method"""
        cannyImage = numpy.invert(cannyImage) 
        
    return cannyImage 
 
if __name__ == "__main__": 
    print("Check input_dir parameters ...")
    assert arguments.input_dir and arguments.input_dir is not None and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "\n ### Please check your input_dir directory (-input_dir)... ###"
    if arguments.input_dir.endswith(os.sep) is False:arguments.input_dir+=os.sep
    assert arguments.export_dir and arguments.export_dir is not None , "Please check your export directory ..."
    if not os.path.exists(arguments.export_dir): os.makedirs(arguments.export_dir, 775) 
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep
    
    ''' Creating export folder structure base on --input_dir structure '''
    copyTree()     
    
    filePaths = collectImageFilePaths()
    
    step = 0
    printProgress(step,len(filePaths))          
    
    for filePath in filePaths:   
        step += 1
        
        image = cv2.imread(filePath)
        if image is None:
            print("Could not process file:",filePath)
            printProgress(step,len(filePaths))
            continue
        
        if str(arguments.method).lower() == "canny":
            image = processCanny(image)
            
        if str(arguments.method).lower() == "sobel":
            image, sobelImageX, sobelImageY  = processSobel(image)
        
        if str(arguments.method).lower() == "laplacian":
            image = processLaplacian(image) 
         
        cv2.imwrite(arguments.export_dir + str(filePath).replace(arguments.input_dir,""),image)
        
        printProgress(step,len(filePaths))
    
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...")
        
     
       
        
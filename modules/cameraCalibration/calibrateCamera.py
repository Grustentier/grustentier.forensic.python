'''
Created on 04.06.2020

@author: grustentier
'''

print("""


   ___                                            ___          _   _   _                   _               
  / __|  __ _   _ __    ___   _ _   __ _   ___   / __|  __ _  | | (_) | |__   _ _   __ _  | |_   ___   _ _ 
 | (__  / _` | | '  \  / -_) | '_| / _` | |___| | (__  / _` | | | | | | '_ \ | '_| / _` | |  _| / _ \ | '_|
  \___| \__,_| |_|_|_| \___| |_|   \__,_|        \___| \__,_| |_| |_| |_.__/ |_|   \__,_|  \__| \___/ |_|  
                                                                                                           
  _                ___                     _                  _     _                                      
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                          
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                         
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                           
         |__/                                                                                              


""")

__version__ = '0.1'

import os
import re 
import cv2
import numpy
import argparse

parser = argparse.ArgumentParser(description='Calibrate camera view by a given reference image')
parser.add_argument('--reference_image_path', help='Path to reference_image_path image.', default='')
parser.add_argument('--image_dir', help='Path to input dir with images to arrange.', default='') 
parser.add_argument('--export_dir', default='', type=str, help='The export/output directory')
parser.add_argument('--method', default='sift', type=str, help='Use sift (default), surf, orb or ecc as method for feature detection')
arguments = parser.parse_args()

FILE_TYPES = ["bmp","jpg","jpeg","png"]

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 755)
        
def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%")     
        
def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.image_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths

def stackImagesECC(reference_image_path,filePathList):
    M = numpy.eye(3, 3, dtype=numpy.float32)
    referenceImage = cv2.imread(reference_image_path,1).astype(numpy.float32) / 255
    referenceImage = cv2.cvtColor(referenceImage,cv2.COLOR_BGR2GRAY)
    
    step = 0
    printProgress(step,len(filePathList))
    
    for file in filePathList:
        step += 1 
        filename = os.path.basename(file)
        image = cv2.imread(file,1).astype(numpy.float32) / 255        
        # Estimate perspective transform
        #criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.001) 
        #warp_matrix = numpy.eye(3, 3, dtype=numpy.float32)
        #s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), referenceImage, M, cv2.MOTION_HOMOGRAPHY,criteria,None)
        _, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), referenceImage, M, cv2.MOTION_HOMOGRAPHY)
        w, h, _ = image.shape
        arrangedImage = cv2.warpPerspective(image, M, (h, w))
        arrangedImage = (arrangedImage*255).astype(numpy.uint8)
        cv2.imwrite(arguments.export_dir+filename.split(".")[0]+"-ECC-.png",arrangedImage)
        printProgress(step,len(filePathList))
            
def arrangeImagesBySIFT(reference_image_path,filePathList):     
    sift = cv2.xfeatures2d.SIFT_create()
    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)   
    referenceImage = cv2.imread(reference_image_path,1)
    referenceImage_kp, referenceImage_des = sift.detectAndCompute(referenceImage, None)    
    
    step = 0
    printProgress(step,len(filePathList))
    
    for file in filePathList:
        step += 1 
        filename = os.path.basename(file)
        image = cv2.imread(file,1)
        imageF = image.astype(numpy.float32) / 255         
        kp, des = sift.detectAndCompute(image, None) 
        # create BFMatcher object
        #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED) 
        # Find matches and sort them in the order of their distance
        matches = matcher.match(referenceImage_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = numpy.float32([referenceImage_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Estimate perspective transformation
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        w, h, _ = imageF.shape
        arrangedImage = cv2.warpPerspective(imageF, M, (h, w))
        arrangedImage = (arrangedImage*255).astype(numpy.uint8)
        cv2.imwrite(arguments.export_dir+filename.split(".")[0]+"-SIFT-.png",arrangedImage)
        printProgress(step,len(filePathList))
            
def arrangeImagesBySURF(reference_image_path,filePathList):     
    surf = cv2.xfeatures2d.SURF_create()
    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)   
    referenceImage = cv2.imread(reference_image_path,1)
    referenceImage_kp, referenceImage_des = surf.detectAndCompute(referenceImage, None)      
    
    step = 0
    printProgress(step,len(filePathList))
    
    for file in filePathList:
        step += 1 
        filename = os.path.basename(file)
        image = cv2.imread(file,1)
        imageF = image.astype(numpy.float32) / 255         
        kp, des = surf.detectAndCompute(image, None) 
        # create BFMatcher object
        #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)     
        # Find matches and sort them in the order of their distance
        matches = matcher.match(referenceImage_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = numpy.float32([referenceImage_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Estimate perspective transformation
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        w, h, _ = imageF.shape
        arrangedImage = cv2.warpPerspective(imageF, M, (h, w))
        arrangedImage = (arrangedImage*255).astype(numpy.uint8)
        cv2.imwrite(arguments.export_dir+filename.split(".")[0]+"-SURF-.png",arrangedImage)
        printProgress(step,len(filePathList))  

def arrangeImagesByORB(reference_image_path,filePathList):
    orb = cv2.ORB_create()   
    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)
    referenceImage = cv2.imread(reference_image_path,1)
    referenceImage_kp = orb.detect(referenceImage, None)
    referenceImage_kp, referenceImage_des = orb.compute(referenceImage, referenceImage_kp)
    
    step = 0
    printProgress(step,len(filePathList))
    
    for file in filePathList:
        step += 1 
        filename = os.path.basename(file)
        image = cv2.imread(file,1)
        imageF = image.astype(numpy.float32) / 255
        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)  
        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)           
        # Find matches and sort them in the order of their distance
        matches = matcher.match(referenceImage_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = numpy.float32([referenceImage_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Estimate perspective transformation
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        w, h, _ = imageF.shape
        arrangedImage = cv2.warpPerspective(imageF, M, (h, w))
        arrangedImage = (arrangedImage*255).astype(numpy.uint8)
        cv2.imwrite(arguments.export_dir+filename.split(".")[0]+"-ORB-.png",arrangedImage)
        printProgress(step,len(filePathList))

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sortByNumericFilename(l):
    l.sort(key=alphanum_key)

if __name__ == "__main__":
    assert arguments.reference_image_path and len(arguments.reference_image_path) > 0 and os.path.exists(arguments.reference_image_path) and os.path.isfile(arguments.reference_image_path), "Please check your reference_image_path image path..."
    assert arguments.image_dir and len(arguments.image_dir) > 0 and os.path.exists(arguments.image_dir) and os.path.isdir(arguments.image_dir), "Please check your directory path with images to calibrate (--image_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory ..."
    if not os.path.exists(arguments.export_dir): os.makedirs(arguments.export_dir, 775) 
    if arguments.image_dir.endswith(os.sep) is False:arguments.image_dir+=os.sep
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep     
    createExportDir(arguments.export_dir)
    assert str(arguments.method).lower() == "sift" or str(arguments.method).lower() == "surf" or str(arguments.method).lower() == "orb" or str(arguments.method).lower() == "ecc"  , "Please select a method for feature detection (sift, surf, orb, ecc) ..."
   
    filePaths = collectImageFilePaths()
    sortByNumericFilename(filePaths)
    
    if str(arguments.method).lower() == "ecc":
        stackImagesECC(arguments.reference_image_path,filePaths)
         
    if str(arguments.method).lower() == "sift":
        arrangeImagesBySIFT(arguments.reference_image_path,filePaths)
         
    if str(arguments.method).lower() == "surf":
        arrangeImagesBySURF(arguments.reference_image_path,filePaths)
         
    if str(arguments.method).lower() == "orb":
        arrangeImagesByORB(arguments.reference_image_path,filePaths)
        
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...")








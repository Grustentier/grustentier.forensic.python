'''
Created on 20.02.2019

@author: localadmin
'''

print("""


  ___                                 ___   _                                     
 |_ _|  _ __    __ _   __ _   ___    | _ ) | |  _  _   _ _   _ _   ___   _ _      
  | |  | '  \  / _` | / _` | / -_)   | _ \ | | | || | | '_| | '_| / -_) | '_|     
 |___| |_|_|_| \__,_| \__, | \___|   |___/ |_|  \_,_| |_|   |_|   \___| |_|       
                      |___/                                                       
  _                ___                     _                  _     _             
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _ 
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|  
         |__/                                                                     


""")

import os
import cv2
import sys
import shutil 
import argparse


parser = argparse.ArgumentParser(description='Code for feature detection.')
parser.add_argument('--input_dir', default='', help='Path to input dir with images to blur.')
parser.add_argument('--export_dir', default='', type=str, help='The export/output directory')
parser.add_argument('--method', default='default', type=str, help='Blurring methods to choose: default, gaussian, median and bilateral')
parser.add_argument('--threshold', default=15, type=int, help='The threshold parameter for setting blurring level')
arguments = parser.parse_args()

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 0o0777)

def blur1(img,threshold = 5):
    kernel = numpy.ones((threshold,threshold),numpy.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def blur(img,threshold = 5):
    return cv2.blur(img,(threshold,threshold)) 

def gaussianBlur(img, threshold = 5):
    return cv2.GaussianBlur(img,(threshold,threshold),0)

def medianBlur(img,threshold = 5):
    return cv2.medianBlur(img,threshold)
 
def bilateralFiltering(img):
    return cv2.bilateralFilter(img,9,75,75)

if __name__ == "__main__":
    methods = ["default","gaussian","median","bilateral"]
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir) ..."
    assert arguments.method in methods,"Please choose one of the following methods:default,gaussian,median or bilateral"
    if not os.path.exists(arguments.export_dir): os.makedirs(arguments.export_dir, 775) 
    if arguments.input_dir.endswith("/") is False:arguments.input_dir+="/"
    if arguments.export_dir.endswith("/") is False:arguments.export_dir+="/"     
    createExportDir(arguments.export_dir)
    
    if os.path.exists(arguments.export_dir):
        try:
            shutil.rmtree(arguments.export_dir)
        except OSError as e:
            print("Error: %s : %s" % (arguments.export_dir, e.strerror))   
 
    shutil.copytree(arguments.input_dir,arguments.export_dir,ignore=ignore_files)
    
    for root, _, files in os.walk(arguments.input_dir):     
        for file in sorted(files):
            exportDirPath = arguments.export_dir + str(root).replace(arguments.input_dir,"")
            if exportDirPath.endswith("/") is False:exportDirPath+="/"  
            if root.endswith("/") is False:root+="/"
            image = cv2.imread(root + file)
            if image is None:
                print("Could not process file:",root + file)
                continue
            
            if arguments.method == "default":
                cv2.imwrite(exportDirPath+file,blur(image,arguments.threshold))
            
            if arguments.method == "gaussian":
                cv2.imwrite(exportDirPath+file,gaussianBlur(image,arguments.threshold))
            
            if arguments.method == "median":
                cv2.imwrite(exportDirPath+file,medianBlur(image,arguments.threshold))            
            
            if arguments.method == "bilateral":
                cv2.imwrite(exportDirPath+file,bilateralFiltering(image)) 
            
            sys.stdout.write(".")
            sys.stdout.flush()
            
    print("FINISHED...")

 

 
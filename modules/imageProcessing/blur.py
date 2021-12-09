'''
Created on 20.02.2019

@author: Grustentier
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
import sys
import cv2
import numpy
import shutil
import argparse

parser = argparse.ArgumentParser(description='Code for multiple image blurring')
parser.add_argument('--input_dir', default=None, help='The input directory with separated video frames (images)')
parser.add_argument('--export_dir', default=None, type=str, help='The export/output directory')
parser.add_argument('--method', default='default', type=str, help='Blurring methods to choose: default, gaussian, median and bilateral')
parser.add_argument('--threshold', default=15, type=int, help='The threshold parameter for setting blurring level')
arguments = parser.parse_args()

FILE_TYPES = ["bmp", "jpg", "jpeg", "png"]


def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def collectImageFilePaths(input_dir):
    filePaths = []
    for root, _, files in os.walk(input_dir): 
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths


def copyTree(input_dir, export_dir):
    if os.path.exists(export_dir):
        try:
            shutil.rmtree(export_dir)
        except OSError as e:
            print("Error: %s : %s" % (export_dir, e.strerror))   
 
    shutil.copytree(input_dir, export_dir, ignore=ignore_files)

    
def printProgress(steps, maximum, name="progress", bar_length=20, width=20): 
    if maximum == 0:
        percent = 1.0
    else:
        percent = float(steps) / maximum
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    # sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent*100))))
    sys.stdout.write("\r{0: <{1}}[{2}]{3}%".format("", 0, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()    
    
    if steps >= maximum: 
        sys.stdout.write('\n\n')


def blur2(img, threshold=5):
    kernel = numpy.ones((threshold, threshold), numpy.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def blur1(img, threshold=5):
    return cv2.blur(img, (threshold, threshold)) 


def gaussianBlur(img, threshold=5):
    return cv2.GaussianBlur(img, (threshold, threshold), 0)


def medianBlur(img, threshold=5):
    return cv2.medianBlur(img, threshold)

 
def bilateralFiltering(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def blur(input_dir, export_dir, method='default', threshold=15):
    methods = ["default", "gaussian", "median", "bilateral"]
    assert input_dir and len(input_dir) > 0 and os.path.exists(input_dir) and os.path.isdir(input_dir), "Please check your input directory (--input_dir)..."
    assert export_dir and len(export_dir) > 0 , "Please check your export directory (--export_dir) ..."
    assert method in methods, "Please choose one of the following methods:default,gaussian,median or bilateral"
    if input_dir.endswith(os.sep) is False:input_dir += os.sep
    if export_dir.endswith(os.sep) is False:export_dir += os.sep     
    
    ''' Creating export folder structure base on --input_dir structure '''
    copyTree(input_dir, export_dir)   
    
    ''' Collecting image file paths '''
    filePaths = collectImageFilePaths(input_dir)    
      
    step = 0
    printProgress(step, len(filePaths))    
    
    for filePath in filePaths: 
        step += 1
                
        image = cv2.imread(filePath)
        if image is None:
            print("Could not process file:", filePath)
            printProgress(step, len(filePaths))
            continue
        
        if method == "default":
            cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), blur1(image, threshold))
        
        if method == "gaussian":
            cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), gaussianBlur(image, threshold))
        
        if method == "median":
            cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), medianBlur(image, threshold))            
        
        if method == "bilateral":
            cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), bilateralFiltering(image)) 
        
        printProgress(step, len(filePaths))
    
    os.system('chmod 777 -R ' + export_dir)
    print("FINISHED...")


if __name__ == "__main__":
    blur(arguments.input_dir, arguments.export_dir, arguments.method, arguments.threshold)    

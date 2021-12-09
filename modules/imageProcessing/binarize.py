'''
Created on 31.01.2019

@author: Grustentier
'''

print("""
  ___                                 ___   _                       _                  
 |_ _|  _ __    __ _   __ _   ___    | _ ) (_)  _ _    __ _   _ _  (_)  ___  ___   _ _ 
  | |  | '  \  / _` | / _` | / -_)   | _ \ | | | ' \  / _` | | '_| | | |_ / / -_) | '_|
 |___| |_|_|_| \__,_| \__, | \___|   |___/ |_| |_||_| \__,_| |_|   |_| /__| \___| |_|  
                      |___/                                                            
  _                ___                     _                  _     _                  
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _      
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|     
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|       
         |__/                                                                          

""")

__version__ = '0.1'

import os
import sys
import cv2
import shutil
import argparse

parser = argparse.ArgumentParser(description='Code for multiple image binarization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default=None, type=str, help='The input directory with separated video frames (images)')
parser.add_argument('--export_dir', default=None, type=str, help='The export directory path') 
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

        
def binarize(input_dir, export_dir):
    assert input_dir and len(input_dir) > 0 and os.path.isdir(input_dir), "Please check your input directory (--input_dir)"
    assert export_dir and len(export_dir) > 0, "Please check your export directory (--export_dir)"
    if input_dir.endswith(os.sep) is False:input_dir += os.sep     
    if export_dir.endswith(os.sep) is False:export_dir += os.sep     
    
    ''' Creating export folder structure base on --input_dir structure '''
    copyTree(input_dir, export_dir)  
    
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
        
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), thresh) 
        
        printProgress(step, len(filePaths))
    
    os.system('chmod 777 -R ' + export_dir)
    print("FINISHED...")

     
if __name__ == "__main__":
    binarize(arguments.input_dir, arguments.export_dir)    

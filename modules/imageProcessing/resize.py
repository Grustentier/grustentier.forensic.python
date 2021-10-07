'''
Created on 22.06.2020

@author: grustentier
'''

import os
import cv2
import shutil
import imutils
import argparse

print("""
  ___                                 ___              _                          
 |_ _|  _ __    __ _   __ _   ___    | _ \  ___   ___ (_)  ___  ___   _ _         
  | |  | '  \  / _` | / _` | / -_)   |   / / -_) (_-< | | |_ / / -_) | '_|        
 |___| |_|_|_| \__,_| \__, | \___|   |_|_\ \___| /__/ |_| /__| \___| |_|          
                      |___/                                                       
  _                ___                     _                  _     _             
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _ 
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|  
         |__/                                                                     

""")

__version__ = '0.1'

parser = argparse.ArgumentParser(description='Code for feature detection.')
parser.add_argument('--input_dir', default='', help='Path to input dir with images to check.')
parser.add_argument('--export_dir', default='', type=str, help='The export/output directory')
parser.add_argument('--height', default=100, type=int, help='The height to resize')
arguments = parser.parse_args() 

FILE_TYPES = ["bmp","jpg","jpeg","png"]

def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.input_dir):  
        if root.endswith("/") is False:root+="/"
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
    for _ in range(0,steps + 1):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%")

if __name__ == "__main__":
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir)..."
    if not os.path.exists(arguments.export_dir): os.makedirs(arguments.export_dir, 775) 
    if arguments.input_dir.endswith("/") is False:arguments.input_dir+="/"
    if arguments.export_dir.endswith("/") is False:arguments.export_dir+="/"     
     
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

        cv2.imwrite(arguments.export_dir + str(filePath).replace(arguments.input_dir,""),imutils.resize(image,height=arguments.height))
        
        printProgress(step,len(filePaths))
    
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...")
    
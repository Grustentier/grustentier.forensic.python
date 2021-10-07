#!/usr/bin/env python3

'''
Changed on 06.11.2019

@author: Grustentier
'''

print("""
  ___                                 ___                      _                         _               
 |_ _|  _ __    __ _   __ _   ___    |   \   ___   ___  __ _  | |_   _  _   _ _   __ _  | |_   ___   _ _ 
  | |  | '  \  / _` | / _` | / -_)   | |) | / -_) (_-< / _` | |  _| | || | | '_| / _` | |  _| / _ \ | '_|
 |___| |_|_|_| \__,_| \__, | \___|   |___/  \___| /__/ \__,_|  \__|  \_,_| |_|   \__,_|  \__| \___/ |_|  
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

parser = argparse.ArgumentParser(description='Converting into a gray scale image',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default='', type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default='', type=str, help='The export/output directory') 
arguments = parser.parse_args() 

FILE_TYPES = ["bmp","jpg","jpeg","png"]
 
def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

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
    
def printProgress(steps,max):
    output = ""
    for i in range(0,steps + 1):
        output +="."
    print("["+output+"]", str(int(round((steps/max)*100,0)))+"%") 
                
if __name__ == "__main__":
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)"
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir)"
    if arguments.input_dir.endswith("/") is False:arguments.input_dir+="/"     
    if arguments.export_dir.endswith("/") is False:arguments.export_dir+="/"  
        
    ''' Creating export folder structure base on --input_dir structure '''
    copyTree()     
    
    filePaths = collectImageFilePaths()
    
    step = 0
    for filePath in filePaths:   
        step += 1
                
        image = cv2.imread(filePath,0)
        if image is None:
            print("Could not process file:",filePath)
            continue      
         
        cv2.imwrite(arguments.export_dir + str(filePath).replace(arguments.input_dir,""),image) 
        
        printProgress(step,len(filePaths))
    
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...")
    
    
    
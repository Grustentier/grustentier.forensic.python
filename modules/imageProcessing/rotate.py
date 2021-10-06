'''
Created on 22.06.2020

@author: grustentier
'''

print("""
  ___                                 ___         _            _                  
 |_ _|  _ __    __ _   __ _   ___    | _ \  ___  | |_   __ _  | |_   ___   _ _    
  | |  | '  \  / _` | / _` | / -_)   |   / / _ \ |  _| / _` | |  _| / _ \ | '_|   
 |___| |_|_|_| \__,_| \__, | \___|   |_|_\ \___/  \__| \__,_|  \__| \___/ |_|     
                      |___/                                                       
  _                ___                     _                  _     _             
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _ 
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|  
         |__/                                                                     

""")

__version__ = '0.1'


import cv2
import os
import sys
import shutil
import imutils
import argparse

parser = argparse.ArgumentParser(description='Code for feature detection.')
parser.add_argument('--input_dir', default='', help='Path to input dir with images to check.')
parser.add_argument('--export_dir', default='', type=str, help='The export/output directory')
parser.add_argument('--angle', default=10, type=int, help='The angle to rotate')
arguments = parser.parse_args()

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 0o0777)

if __name__ == "__main__":
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir) ..."
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
            
            rotated = imutils.rotate_bound(image, arguments.angle)
            cv2.imwrite(exportDirPath+file,rotated)
            sys.stdout.write(".")
            sys.stdout.flush()
            
    print("FINISHED...")
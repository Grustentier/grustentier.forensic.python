#!/usr/bin/env python3

'''
Changed on 06.11.2019

@author: Grustentier
'''

print("""
  ___                                 ___                               ___                    _          _     _                                                                                                                                                                                             
 |_ _|  _ __    __ _   __ _   ___    / __|  _  _   _ __   ___   _ _    | _ \  ___   ___  ___  | |  _  _  | |_  (_)  ___   _ _                                                                                                                                                                                 
  | |  | '  \  / _` | / _` | / -_)   \__ \ | || | | '_ \ / -_) | '_|   |   / / -_) (_-< / _ \ | | | || | |  _| | | / _ \ | ' \                                                                                                                                                                                
 |___| |_|_|_| \__,_| \__, | \___|   |___/  \_,_| | .__/ \___| |_|     |_|_\ \___| /__/ \___/ |_|  \_,_|  \__| |_| \___/ |_||_|                                                                                                                                                                               
                      |___/                       |_|                                                                                                                                                                                                                                                         
  _               _      _     _                 _     __    __         _   _     _             _                               __  _      _                _           __  _                                                                                                 _          _     _              
 | |__   _  _    | |_   | |_  | |_   _ __   ___ (_)   / /   / /  __ _  (_) | |_  | |_    _  _  | |__       __   ___   _ __     / / (_)  __| |  ___   __ _  | |  ___    / / (_)  _ __    __ _   __ _   ___   ___   ___  _  _   _ __   ___   _ _   ___   _ _   ___   ___  ___  | |  _  _  | |_  (_)  ___   _ _  
 | '_ \ | || |   | ' \  |  _| |  _| | '_ \ (_-<  _   / /   / /  / _` | | | |  _| | ' \  | || | | '_ \  _  / _| / _ \ | '  \   / /  | | / _` | / -_) / _` | | | / _ \  / /  | | | '  \  / _` | / _` | / -_) |___| (_-< | || | | '_ \ / -_) | '_| |___| | '_| / -_) (_-< / _ \ | | | || | |  _| | | / _ \ | ' \ 
 |_.__/  \_, |   |_||_|  \__|  \__| | .__/ /__/ (_) /_/   /_/   \__, | |_|  \__| |_||_|  \_,_| |_.__/ (_) \__| \___/ |_|_|_| /_/   |_| \__,_| \___| \__,_| |_| \___/ /_/   |_| |_|_|_| \__,_| \__, | \___|       /__/  \_,_| | .__/ \___| |_|         |_|   \___| /__/ \___/ |_|  \_,_|  \__| |_| \___/ |_||_|
         |__/                       |_|                         |___/                                                                                                                         |___/                          |_|                                                                              

""")

'''
Image Super Resolution work-around based on https://github.com/idealo/image-super-resolution
'''

__version__ = '0.1'

import os
import cv2
import sys
import numpy
import shutil
import argparse
from PIL import Image
from ISR.models import RDN

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default='', type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default='', type=str, help='The export directory path')
parser.add_argument('--model', default='small', type=str, help='Select small or large for choosing prediction model') 
arguments = parser.parse_args() 

IMAGE_FILE_TYPES = ["bmp", "png", "jpg", "jpeg"]


def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def copyTree():
    if os.path.exists(arguments.export_dir):
        try:
            shutil.rmtree(arguments.export_dir)
        except OSError as e:
            print("Error: %s : %s" % (arguments.export_dir, e.strerror))   
 
    shutil.copytree(arguments.input_dir, arguments.export_dir, ignore=ignore_files)


def collectImageFilePaths(dir):
    filePaths = []
    for root, _, files in os.walk(dir): 
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in IMAGE_FILE_TYPES])
    return filePaths 


def createExportDirs(exportDirPath):
    if os.path.exists(exportDirPath): return
    os.makedirs(str(exportDirPath), 755)

        
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


def increase(input_dir, export_dir, model): 
    assert input_dir and len(input_dir) > 0 and os.path.isdir(input_dir), "Please check your input directory (--input_dir)..."
    if str(input_dir).endswith(os.sep) is False:input_dir += os.sep
    assert export_dir and len(export_dir) > 0, "Please check your input directory (--export_dir)..."
    if str(export_dir).endswith(os.sep) is False:export_dir += os.sep
    
    ''' Creating export folder structure based on --input_dir structure '''
    copyTree() 
    
    createExportDirs(export_dir)
    
    filePaths = collectImageFilePaths(input_dir)
          
    step = 0
    for filePath in filePaths: 
        step += 1
        image = cv2.imread(filePath)  
        
        if image is None:
            print("Could not process image file:", filePath)
            printProgress(step, len(filePaths))
            continue 
        
        rdn = RDN(weights='psnr-' + model)
        sr_img = rdn.predict(image, by_patch_of_size=50)
        cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), sr_img)
        
        printProgress(step, len(filePaths))


if __name__ == "__main__":
    increase(arguments.input_dir, arguments.export_dir, arguments.model)


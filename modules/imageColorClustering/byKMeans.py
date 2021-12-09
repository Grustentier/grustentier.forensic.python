'''
Created on 16.04.2020

@author: Grustentier
'''

print("""
  ___                                  ___         _                  ___   _               _                 _               
 |_ _|  _ __    __ _   __ _   ___     / __|  ___  | |  ___   _ _     / __| | |  _  _   ___ | |_   ___   _ _  (_)  _ _    __ _ 
  | |  | '  \  / _` | / _` | / -_)   | (__  / _ \ | | / _ \ | '_|   | (__  | | | || | (_-< |  _| / -_) | '_| | | | ' \  / _` |
 |___| |_|_|_| \__,_| \__, | \___|    \___| \___/ |_| \___/ |_|      \___| |_|  \_,_| /__/  \__| \___| |_|   |_| |_||_| \__, |
                      |___/                                                                                             |___/ 
              _                   _  __  __  __                                                                               
  _  _   ___ (_)  _ _    __ _    | |/ / |  \/  |  ___   __ _   _ _    ___                                                     
 | || | (_-< | | | ' \  / _` |   | ' <  | |\/| | / -_) / _` | | ' \  (_-<                                                     
  \_,_| /__/ |_| |_||_| \__, |   |_|\_\ |_|  |_| \___| \__,_| |_||_| /__/                                                     
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
import numpy
import shutil
import argparse

parser = argparse.ArgumentParser(description='Code for multiple image clustering', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default=None, type=str, help='The input directory with separated video frames (images)')
parser.add_argument('--export_dir', default=None, type=str, help='The export/output directory')
parser.add_argument('--clusters', default=2, type=int, help='Number of clusters') 
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


def cluster(img, clusters): 
    Z = img.reshape((-1, 3)) 
    # convert to np.float32
    Z = numpy.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    _, label, center = cv2.kmeans(Z, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = numpy.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2 


def byKMeans(input_dir, export_dir, clusters=2):
    assert input_dir and len(input_dir) > 0 and os.path.exists(input_dir) and os.path.isdir(input_dir), "Please check your input directory (--input_dir)..."
    assert export_dir and len(export_dir) > 0 , "Please check your export directory (--export_dir) ..."
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
         
        cv2.imwrite(export_dir + str(filePath).replace(input_dir, ""), cluster(image, clusters))
        
        printProgress(step, len(filePaths))
    
    os.system('chmod 777 -R ' + export_dir)
    print("FINISHED...") 

    
if __name__ == "__main__":
    byKMeans(arguments.input_dir, arguments.export_dir, arguments.clusters)    

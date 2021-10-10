'''
Created on 16.04.2020

@author: grustentier
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
import cv2
import numpy
import shutil
import argparse

parser = argparse.ArgumentParser(description='Increases the contrast of a image by two different applied methods.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default='', type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default='', type=str, help='The export directory path')
parser.add_argument('--cluster', default=2, type=int, help='Number of clusters') 
arguments = parser.parse_args() 

FILE_TYPES = ["bmp","jpg","jpeg","png"]

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

def cluster(img): 
    Z = img.reshape((-1,3)) 
    # convert to np.float32
    Z = numpy.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    _,label,center=cv2.kmeans(Z,arguments.cluster,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = numpy.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2 
    
if __name__ == "__main__":
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir) ..."
    if arguments.input_dir.endswith(os.sep) is False:arguments.input_dir+=os.sep
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
         
        cv2.imwrite(arguments.export_dir + str(filePath).replace(arguments.input_dir,""),cluster(image))
        
        printProgress(step,len(filePaths))
    
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...") 
    
#!/usr/bin/env python3

'''
Changed on 09.12.2019

@author: Grustentier
'''

print("""


  ___   _                            __  __          _          _      _                           
 / __| | |_    __ _   _ __   ___    |  \/  |  __ _  | |_   __  | |_   (_)  _ _    __ _             
 \__ \ | ' \  / _` | | '_ \ / -_)   | |\/| | / _` | |  _| / _| | ' \  | | | ' \  / _` |            
 |___/ |_||_| \__,_| | .__/ \___|   |_|  |_| \__,_|  \__| \__| |_||_| |_| |_||_| \__, |            
                     |_|                                                         |___/             
              _                   _  _                __  __                              _        
  _  _   ___ (_)  _ _    __ _    | || |  _  _   ___  |  \/  |  ___   _ __    ___   _ _   | |_   ___
 | || | (_-< | | | ' \  / _` |   | __ | | || | |___| | |\/| | / _ \ | '  \  / -_) | ' \  |  _| (_-<
  \_,_| /__/ |_| |_||_| \__, |   |_||_|  \_,_|       |_|  |_| \___/ |_|_|_| \___| |_||_|  \__| /__/
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
import pandas
import seaborn 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Image similarity by local alignment using Smith-Waterman algorithm.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default='', type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default='', type=str, help='The export directory path')  
arguments = parser.parse_args() 

FILE_TYPES = ["bmp","jpg","jpeg","png"]

def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.input_dir):  
        if root.endswith("/") is False:root+="/"
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 0o0777)

def createClusterMap(matrix,cols,names,exportFilePath = None,show = False): 
    dataframe = pandas.DataFrame(data=matrix, index=cols,columns=names)
    seaborn.set(font_scale=1)
    h = seaborn.clustermap(dataframe,cmap = "coolwarm")
     
    if exportFilePath is not None:
        plt.savefig(exportFilePath,dpi=100)
    else:
        plt.show() 
    if show:
        plt.show() 
        
    return h

def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%") 

def analyzeByHuMoments(imagePaths):
    d1s = {}
    d2s = {}
    d3s = {}
    
    comatrix_d1 = []  
    comatrix_d2 = []
    comatrix_d3 = []  

    checkedPairs = []
    
    printProgress(0,len(imagePaths))
    step = 0
    for i in range(0,len(imagePaths)): 
        step += 1
        imageName1 = imagePaths[i][imagePaths[i].rfind("/")+1:]
        im1 = cv2.imread(imagePaths[i],cv2.IMREAD_GRAYSCALE)
        matrixEntry_d1 = [] 
        matrixEntry_d2 = []
        matrixEntry_d3 = []
        for j in range(0,len(imagePaths)):           
            imageName2 = imagePaths[j][imagePaths[j].rfind("/")+1:]        
            parentFolderName = os.path.abspath(os.path.join(imagePaths[j], os.pardir))
            parentFolderName = os.path.basename(parentFolderName)
            imageName2 = parentFolderName + "->" + imageName2 
        
            im2 = cv2.imread(imagePaths[j],cv2.IMREAD_GRAYSCALE)
 
            d1 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I1,0)
            d2 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)
            d3 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I3,0) 
            
            if i != j and str(i)+"-"+str(j) not in checkedPairs:
                d1s[imageName1 + " " + imageName2] = round(d1*100,3)
                d2s[imageName1 + " " + imageName2] = round(d2*100,3)
                d3s[imageName1 + " " + imageName2] = round(d3*100,3)
                
            checkedPairs.append(str(i)+"-"+str(j))
            checkedPairs.append(str(j)+"-"+str(i))
            
            matrixEntry_d1.append(round(d1*100,3))         
            matrixEntry_d2.append(round(d2*100,3))
            matrixEntry_d3.append(round(d3*100,3))
        
        comatrix_d1.append(matrixEntry_d1)    
        comatrix_d2.append(matrixEntry_d2)
        comatrix_d3.append(matrixEntry_d3)  
        
        printProgress(step,len(imagePaths))
            
    sorted_d1s = sorted(d1s.items(), key=lambda kv: kv[1])
    sorted_d2s = sorted(d2s.items(), key=lambda kv: kv[1])
    sorted_d3s = sorted(d3s.items(), key=lambda kv: kv[1])
    
    print("Distance CONTOURS_MATCH_I1")
    for d1s in sorted_d1s: print(d1s)
    print("\nDistance CONTOURS_MATCH_I2")
    for d2s in sorted_d2s: print(d2s)
    print("\nDistance CONTOURS_MATCH_I3")
    for d3s in sorted_d3s: print(d3s)
    
    cols = [str(i) for i in range(0,len(imagePaths))]
    names = [imagePaths[i].split("/")[-1].split(".")[0] for i in range(0,len(imagePaths))]  
    createClusterMap(comatrix_d1,cols,names,arguments.export_dir+"clusterMap_CONTOURS_MATCH_I1.png")   
    createClusterMap(comatrix_d2,cols,names,arguments.export_dir+"clusterMap_CONTOURS_MATCH_I2.png")
    createClusterMap(comatrix_d3,cols,names,arguments.export_dir+"clusterMap_CONTOURS_MATCH_I3.png")

if __name__ == "__main__": 
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)"
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir)"
    if arguments.input_dir.endswith("/") is False:arguments.input_dir+="/"
    if arguments.export_dir.endswith("/") is False:arguments.export_dir+="/"
    createExportDir(arguments.export_dir) 
    
    filePaths = collectImageFilePaths() 
    
    analyzeByHuMoments(filePaths) 
    print("FINISHED...")
    

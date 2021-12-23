#!/usr/bin/env python3

'''
Changed on 09.12.2019

@author: Grustentier
'''


print("""


   ___         _                 _  _   _        _                                                ___                                            _     _              
  / __|  ___  | |  ___   _ _    | || | (_)  ___ | |_   ___   _ _   __ _   _ _   __ _   _ __      / __|  ___   _ __    _ __   __ _   _ _   __ _  | |_  (_)  ___   _ _  
 | (__  / _ \ | | / _ \ | '_|   | __ | | | (_-< |  _| / _ \ | '_| / _` | | '_| / _` | | '  \    | (__  / _ \ | '  \  | '_ \ / _` | | '_| / _` | |  _| | | / _ \ | ' \ 
  \___| \___/ |_| \___/ |_|     |_||_| |_| /__/  \__| \___/ |_|   \__, | |_|   \__,_| |_|_|_|    \___| \___/ |_|_|_| | .__/ \__,_| |_|   \__,_|  \__| |_| \___/ |_||_|
                                                                  |___/                                              |_|                                              
  _                ___                     _                  _     _                                                                                                 
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                                                                                     
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                                                                                    
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                                                                                      
         |__/                                                                                                                                                         


""")

__version__ = '0.1'

import os
import cv2
import sys
import pandas
import seaborn
import argparse 
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Image similarity by local alignment using Smith-Waterman algorithm.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default=None, type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default=None, type=str, help='The export directory path')  
arguments = parser.parse_args()  

FILE_TYPES = ["bmp", "jpg", "jpeg", "png"] 

OPENCV_METHODS = list()
OPENCV_METHODS.append(("Correlation-(winner=max)", cv2.HISTCMP_CORREL, "max"))
OPENCV_METHODS.append(("Intersection-(winner=max)", cv2.HISTCMP_INTERSECT, "max"))
OPENCV_METHODS.append(("Chi-Squared-(winner=min)", cv2.HISTCMP_CHISQR, "min"))
OPENCV_METHODS.append(("Hellinger-(winner=min)", cv2.HISTCMP_BHATTACHARYYA, "min"))
OPENCV_METHODS.append(("Kullback-Leibler-(winner=min)", cv2.HISTCMP_KL_DIV, "min"))


def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def collectImageFilePaths(input_dir):
    filePaths = []
    for root, _, files in os.walk(input_dir): 
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths


def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 0o0777)
 

def getHistogram(image):
    '''
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges 
    channels = [0, 1]
    hist_test = cv2.calcHist([image], channels, None, histSize, ranges, accumulate=False)
    '''    
    hist_test = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_test 


def getHistograms(filePaths): 
    # return  [getHistogram(cv2.imread(filePaths[i], cv2.COLOR_BGR2HSV)) for i in range(0, len(filePaths))] 
    return  [getHistogram(cv2.imread(filePaths[i], 0)) for i in range(0, len(filePaths))]
        

def printProgress(steps, maximum, name="progress", bar_length=20, width=20): 
    if maximum == 0:
        percent = 1.0
    else:
        percent = float(steps) / maximum
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}}[{2}]{3}%".format("", 0, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()    
    
    if steps >= maximum: 
        sys.stdout.write('\n\n')

        
def createClusterMap(matrix, cols, names, exportFilePath=None, show=False): 
    dataframe = pandas.DataFrame(data=matrix, index=cols, columns=names)
    seaborn.set(font_scale=1)
    h = seaborn.clustermap(dataframe, cmap="coolwarm") 
    
    if exportFilePath is not None:
        plt.savefig(exportFilePath, dpi=100)
    else:
        plt.show() 
    if show:
        plt.show() 
        
    return h        


def compare(filePaths): 
    historgrams = getHistograms(filePaths)    
    printProgress(0, len(OPENCV_METHODS) - 1)
    for methodId in range(0, len(OPENCV_METHODS)):
        method = OPENCV_METHODS[methodId][1]
        methodName = str(OPENCV_METHODS[methodId][0]).split("-")[0]   
        comatrix = []        
        for h_i in historgrams: 
            matrixEntry = [] 
            for h_j in historgrams:
                histogramResult = cv2.compareHist(h_i, h_j, method)
                matrixEntry.append(histogramResult)            
            comatrix.append(matrixEntry)            
        cols = [str(i) for i in range(0, len(filePaths))]
        names = [filePaths[i].split(os.sep)[-1].split(".")[0] for i in range(0, len(filePaths))]  
        createClusterMap(comatrix, cols, names, arguments.export_dir + os.sep + methodName + ".png")  
        printProgress(methodId, len(OPENCV_METHODS) - 1)   


def compareColors(input_dir, export_dir):
    assert input_dir and len(input_dir) > 0 and os.path.exists(input_dir) and os.path.isdir(input_dir), "Please check your input directory (--input_dir)..."
    assert export_dir and len(export_dir) > 0 , "Please check your export directory (--export_dir)..."    
    if input_dir.endswith(os.sep) is False:input_dir += os.sep
    if export_dir.endswith(os.sep) is False:export_dir += os.sep     
    createExportDir(arguments.export_dir)      
    compare(collectImageFilePaths(input_dir))     
    print("FINISHED...")


if __name__ == "__main__": 
    compareColors(arguments.input_dir, arguments.export_dir)
    

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
  _               ___                _     _        __  __                             ___    _        _                            
 | |__   _  _    | __|  __ _   _ _  | |_  | |_     |  \/  |  ___  __ __  ___   _ _    |   \  (_)  ___ | |_   __ _   _ _    __   ___ 
 | '_ \ | || |   | _|  / _` | | '_| |  _| | ' \    | |\/| | / _ \ \ V / / -_) | '_|   | |) | | | (_-< |  _| / _` | | ' \  / _| / -_)
 |_.__/  \_, |   |___| \__,_| |_|    \__| |_||_|   |_|  |_| \___/  \_/  \___| |_|     |___/  |_| /__/  \__| \__,_| |_||_| \__| \___|
         |__/                                                                                                                       
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
import numpy as np
from skimage import metrics
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats import wasserstein_distance

parser = argparse.ArgumentParser(description='Image similarity by local alignment using Smith-Waterman algorithm.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', default='/home/grustentier/Bilder/data_small', type=str, help='The input directory with separated frames/images')
parser.add_argument('--export_dir', default='/home/grustentier/Bilder/', type=str, help='The export directory path')  
arguments = parser.parse_args() 

height = 1024
width = 1024 

FILE_TYPES = ["bmp","jpg","jpeg","png"]

def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.input_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
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

def get_img(path, norm_size=True, norm_exposure=False):
    '''
      Prepare an image for image processing tasks
      '''
    # flatten returns a 2d grayscale array
    # img = imread(path, flatten=True).astype(int)  
    img = cv2.imread(path,0).astype(int)  
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    if norm_size:
        img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img


def get_histogram(img):
    '''
      Get the histogram of an image. For an 8-bit, grayscale image, the
      histogram will be a 256 unit vector in which the nth value indicates
      the percent of the pixels in the image with the given darkness level.
      The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w) 


def normalize_exposure(img):
    '''
      Normalize the exposure of an image.
      '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
    '''
      Measure the Earth Mover's distance between two images
      @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
      @returns:
        TODO
      '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
  
    return wasserstein_distance(hist_a, hist_b)


def structural_sim(path_a, path_b):
    '''
      Measure the structural similarity between two images
      @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
      @returns:
        {float} a float {-1:1} that measures structural similarity
          between the input images
      '''
    img_a = get_img(path_a)
    img_b = get_img(path_b)
    sim, _ = metrics.structural_similarity(img_a, img_b, full=True)
    return sim


def pixel_sim(path_a, path_b):
    '''
      Measure the pixel-level similarity between two images
      @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
      @returns:
        {float} a float {-1:1} that measures structural similarity
          between the input images
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def sift_sim(path_a, path_b):
    '''
    Use SIFT features to measure image similarity
    @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
    @returns:
    TODO
    '''
    # initialize the sift feature detector
    orb = cv2.ORB_create()

    # get the images
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    # find the keypoints and descriptors with SIFT
    _, desc_a = orb.detectAndCompute(img_a, None)
    _, desc_b = orb.detectAndCompute(img_b, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)
  
def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%") 
    
def emdResults2Console(imagePaths):
    emds = {}
    structuralSims = {}
    pixelSims = {}
    
    comatrix_emd = []
    comatrix_structuralSim = []   
    comatrix_pixelSim = [] 
    
    checkedPairs = []
    
    printProgress(0,len(imagePaths))
    step = 0    
    for i in range(0,len(imagePaths)): 
        step += 1
        iPath = imagePaths[i]
        imageName1 = iPath[iPath.rfind(os.sep)+1:] 
        matrixEntry_emd = [] 
        matrixEntry_structuralSim = []
        matrixEntry_pixelSim = []
        for j in range(0,len(imagePaths)):  
            imageName2 = imagePaths[j][imagePaths[j].rfind(os.sep)+1:]      
            parentFolderName = os.path.abspath(os.path.join(imagePaths[j], os.pardir))
            parentFolderName = os.path.abspath(os.path.join(parentFolderName, os.pardir))
            parentFolderName = os.path.basename(parentFolderName)
            imageName2 = parentFolderName + "->" + imageName2 
            # get the similarity values
            structuralSim = structural_sim(iPath, imagePaths[j])
            matrixEntry_structuralSim.append(structuralSim)
            pixelSim = pixel_sim(iPath, imagePaths[j])
            matrixEntry_pixelSim.append(pixelSim)
            #siftSim = sift_sim(iPath, imagePaths[j])
            emd = earth_movers_distance(iPath, imagePaths[j]) 
            matrixEntry_emd.append(emd*100)  
            
            if i != j and str(i)+"-"+str(j) not in checkedPairs:
                emds[imageName1 + " " + imageName2]= emd*100 #round(emd*100,3)
                structuralSims[imageName1 + " " + imageName2] = structuralSim*100 #round(structuralSim*100,3)
                pixelSims[imageName1 + " " + imageName2]= pixelSim*100 #round(pixelSim*100,3)
                
            checkedPairs.append(str(i)+"-"+str(j))
            checkedPairs.append(str(j)+"-"+str(i))
        
        comatrix_emd.append(matrixEntry_emd)  
        comatrix_structuralSim.append(matrixEntry_structuralSim)  
        comatrix_pixelSim.append(matrixEntry_pixelSim) 
        
        printProgress(step,len(imagePaths))
            
    sorted_emds = sorted(emds.items(), key=lambda kv: kv[1])
    sorted_structuralSims = sorted(structuralSims.items(), key=lambda kv: kv[1])
    sorted_pixelSims = sorted(pixelSims.items(), key=lambda kv: kv[1])
    
    print("EMDs")
    for emd in sorted_emds:print(emd)
    print("\nstructuralSims")
    for structuralSim in reversed(sorted_structuralSims):print(structuralSim)
    print("\npixelSims")    
    for pixelSim in sorted_pixelSims:print(pixelSim)     
    
    cols = [str(i) for i in range(0,len(imagePaths))]
    names = [imagePaths[i].split(os.sep)[-1].split(".")[0] for i in range(0,len(imagePaths))]  
    createClusterMap(comatrix_emd,cols,names,arguments.export_dir+"earth-mover-distance.png")         
    createClusterMap(comatrix_pixelSim,cols,names,arguments.export_dir+"pixel-similarity.png")
    createClusterMap(comatrix_structuralSim,cols,names,arguments.export_dir+"structural-similarity.png") 

if __name__ == "__main__": 
    assert arguments.input_dir and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir)..."    
    if arguments.input_dir.endswith(os.sep) is False:arguments.input_dir+=os.sep
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep 
    
    createExportDir(arguments.export_dir)  
    
    filePaths = collectImageFilePaths()        
      
    emdResults2Console(filePaths) 
    
    print("FINISHED...")
                 
    
            
                
         

    
    
    
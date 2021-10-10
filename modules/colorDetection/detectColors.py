'''
Created on 11.11.2020

@author: grustentier
'''
import os    
import cv2 
import shutil 
import argparse 


print("""


   ___         _                     _  _   _        _                                             ___   _             _             
  / __|  ___  | |  ___   _ _   ___  | || | (_)  ___ | |_   ___   __ _   _ _   __ _   _ __    ___  | __| (_)  _ _    __| |  ___   _ _ 
 | (__  / _ \ | | / _ \ | '_| |___| | __ | | | (_-< |  _| / _ \ / _` | | '_| / _` | | '  \  |___| | _|  | | | ' \  / _` | / -_) | '_|
  \___| \___/ |_| \___/ |_|         |_||_| |_| /__/  \__| \___/ \__, | |_|   \__,_| |_|_|_|       |_|   |_| |_||_| \__,_| \___| |_|  
                                                                |___/                                                                
  _                ___                     _                  _     _                                                                
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                                                    
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                                                   
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                                                     
         |__/                                                                                                                        


""") 

'''
This script allows you to search for specific areas in an image whose color histograms have the greatest similarity to known color histograms.
In the forensic context, this can be very helpful to quantitatively assess visually cognitively perceivable colors.
'''

parser = argparse.ArgumentParser(description='Code for histogram comparation between images.')
parser.add_argument('--image_path', default="."+os.sep+"testdata"+os.sep+"color_spectrum.png", type=str, help='Path to image where color histograms have to be find.')
parser.add_argument('--raster_image_dir', default="."+os.sep+"testdata"+os.sep+"raster"+os.sep+"", type=str, help='Path to input dir with sub dirs representing color classes with raster images.')
parser.add_argument('--export_dir', default="."+os.sep+"testdata"+os.sep+"results"+os.sep+"", type=str, help='The export directory')
parser.add_argument('--raster_window_size', default=10, type=int, help='The size for flying window during the rastering process of --image_path.')
parser.add_argument('--raster_window_step', default=10, type=int, help='The window moving step (x to xn and y to yn) during the raster process of --image_path.')
parser.add_argument('--color_legend', default=False, type=str, help='The color legend representing distance value ranges')
arguments = parser.parse_args()

FILE_TYPES = ["bmp","jpg","jpeg","png"]
K_WINNERS=[3,6,12,24,48]

COLORS2RANGE = {}
COLORS2RANGE["0.0-0.599"]=[0,0,0]
COLORS2RANGE["0.60-0.699"]=[79,79,79]
COLORS2RANGE["0.70-0.799"]=[179,179,179]
COLORS2RANGE["0.80-0.899"]=[222,222,222]
COLORS2RANGE["0.90-1.00"]=[255,255,255]

SORTED_COLORS = [[255,255,255],[222,222,222],[179,179,179],[79,79,79],[0,0,0]]

OPENCV_METHODS = list()
OPENCV_METHODS.append(("Correlation-(winner=max)", cv2.HISTCMP_CORREL,"max"))
OPENCV_METHODS.append(("Intersection-(winner=max)", cv2.HISTCMP_INTERSECT,"max"))
OPENCV_METHODS.append(("Chi-Squared-(winner=min)", cv2.HISTCMP_CHISQR,"min"))
OPENCV_METHODS.append(("Hellinger-(winner=min)", cv2.HISTCMP_BHATTACHARYYA,"min"))
OPENCV_METHODS.append(("Kullback-Leibler-(winner=min)", cv2.HISTCMP_KL_DIV,"min"))

def getRasterClasses():
    classes = {}
    for root, _, files in os.walk(arguments.raster_image_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        for file in files:
            clazz = str(root + file).split(os.sep)[-2]
            if clazz not in classes.keys():classes[clazz] = []
            if str(file).split(".")[-1].lower() in FILE_TYPES:
                classes[clazz].append(root + file) 
    return classes 

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        os.makedirs(exportDirPath, 0o0777)

def boolean_string(s):
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'

def getHistogram(hsvImage):
    ## [Using 50 bins for hue and 60 for saturation]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    ## [Using 50 bins for hue and 60 for saturation]
    hist_test = cv2.calcHist([hsvImage], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_test 

def byValueKey(item): return item["value"] 

def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%") 
    
def getRasterInformation(image,raster = 20,windowSteps = 1):
    rasterInformation = [];
    (h, w) = image.shape[:2]
    for startX in range(0,w,windowSteps):
        for startY in range(0,h,windowSteps): 
            rasterInformation.append([startX,startX+raster,startY,startY+raster])
    return rasterInformation

def cropImage(img, startX,endX,startY,endY,THRESHOLD = 50):   
    try:
        return img[startY-THRESHOLD: endY+THRESHOLD,startX-THRESHOLD: endX+THRESHOLD]
    except:
        try:
            return img[startY: endY,startX: endX] 
        except:
            return img 

def compareHistograms(): 
    for clazz in RASTER_CLASSES.keys():
        currentResult = []
        for rasterImageFilePath in RASTER_CLASSES[clazz]:
            sf = cv2.imread(rasterImageFilePath)
            sf = cv2.cvtColor(sf, cv2.COLOR_BGR2HSV)
            sfHistogram = getHistogram(sf)     
        
            for rasterInfo in IMAGE_2_FIND_COLORS_RASTER_INFORMATION:
                rasterImage = cropImage(IMAGE_2_FIND_COLORS.copy(), rasterInfo[0], rasterInfo[1], rasterInfo[2], rasterInfo[3], 0)
                rasterImage = cv2.cvtColor(rasterImage, cv2.COLOR_BGR2HSV)
                rasterHistogram = getHistogram(rasterImage)        
            
                histogramResult = cv2.compareHist(sfHistogram, rasterHistogram, methodId)
                currentResult.append({"value":histogramResult,"area":rasterInfo,"rasterImage":sf,"rasterClass":clazz})
        
        if len(currentResult) == 0:
            print("Could not find any histogram matches ...")
            print("Try to change ratio parameter for histogram similarity and/or raster window size ...")                        
         
        if OPENCV_METHODS[methodId][2] == "max":
            currentResult = sorted(currentResult, key=lambda k: k['value'], reverse=True) 
        else:
            currentResult = sorted(currentResult, key=lambda k: k['value'], reverse=False) 
         
        print("\nMatch results for class:"+clazz+" and k="+str(k_winners))
        exportResult(currentResult[:k_winners],methodId) 
        currentResult = [] 

def getColorFromRange(value):    
    for rangeKey in COLORS2RANGE.keys():
        lower = float(rangeKey.split("-")[0])
        upper = float(rangeKey.split("-")[1])
        if value >= lower and value <= upper:
            return COLORS2RANGE[rangeKey]
    return [0,0,0]

def exportResult(result,methodId):    
    df_copy = IMAGE_2_FIND_COLORS.copy() 
    
    kIndex = 0
    print("k\tdist\tstart_x\tstart_y\tend_x\tend_y\tclass")        
    for topK in result:  
        kIndex+=1  
        startX = topK["area"][0]
        endX = topK["area"][1]
        startY = topK["area"][2]
        endY = topK["area"][3]
        cv2.rectangle(df_copy, (int(startX), int(startY)), (int(endX), int(endY)), getColorFromRange(topK["value"]), 2)
        print(str(kIndex)+".\t"+str(round(topK["value"],3))+"\t"+str(startX)+"\t"+str(startY)+"\t"+str(endX)+"\t"+str(endY)+"\t"+topK["rasterClass"])
        
    if boolean_string(arguments.color_legend) is True:
        frame_height, frame_width = df_copy.shape[:2]  
        colorSegmentWidth = int(round(frame_width / len(SORTED_COLORS),0))
        last_x = 0
        for sortedColor in SORTED_COLORS:
            for _ in range(1,colorSegmentWidth):
                for y in range(1,10):
                    df_copy[frame_height-y,last_x] = sortedColor
                last_x = last_x + 1
            
    cv2.imwrite(arguments.export_dir+OPENCV_METHODS[methodId][0].split("-")[0]+os.sep+str(arguments.image_path.split(os.sep)[-1].split(".")[0])+"_k:"+str(k_winners)+"_windowStep:"+str(arguments.raster_window_step)+"_windowSize:"+str(arguments.raster_window_size)+"_class:"+topK["rasterClass"]+"_"+".png",df_copy)
 
def checkParameters():
    print("Check input parameters ...")
    assert arguments.raster_image_dir and arguments.raster_image_dir is not None and os.path.exists(arguments.raster_image_dir) and os.path.isdir(arguments.raster_image_dir), "\n ### Please check your input directory (--raster_image_dir)..."
    if arguments.raster_image_dir.endswith(os.sep) is False:arguments.raster_image_dir+=os.sep
    assert len([d for d in os.scandir(arguments.raster_image_dir) if d.is_dir() is True]) > 0, "No class specific sub dirs in --raster_image_dir has been found. Copy your raster images into a subdirectory, where each subdirectory represents a class."
    assert arguments.image_path and arguments.image_path is not None and os.path.exists(arguments.image_path) and os.path.isfile(arguments.image_path), "\n ### Please check your second input image (--image_path) ..."
    assert arguments.export_dir and arguments.export_dir is not None , "Please check your export directory (--export_dir)..."
    if not os.path.exists(arguments.export_dir): os.makedirs(arguments.export_dir, 775) 
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep
    assert arguments.raster_window_size and arguments.raster_window_size is not None and arguments.raster_window_size > 0,"Please check your window size (--raster_window_size)..."

if __name__ == "__main__":  
    checkParameters() 

    shutil.rmtree(arguments.export_dir)
    for methodId in range(0,len(OPENCV_METHODS)):
        methodName = str(OPENCV_METHODS[methodId][0]).split("-")[0]
        createExportDir(arguments.export_dir+methodName)  
        os.system('chmod 777 -R ' + arguments.export_dir+methodName+os.sep) 
    
    IMAGE_2_FIND_COLORS = cv2.imread(arguments.image_path)  
    IMAGE_2_FIND_COLORS_RASTER_INFORMATION = getRasterInformation(IMAGE_2_FIND_COLORS, arguments.raster_window_size,arguments.raster_window_step)
    RASTER_CLASSES = getRasterClasses() 

    step = 0
    printProgress(step,len(OPENCV_METHODS)+len(K_WINNERS)) 
    for k in K_WINNERS:
        k_winners = k        
        for methodId in range(0,len(OPENCV_METHODS)):
            step+=1
             
            print("\nSeach similar histograms using method:",str(OPENCV_METHODS[methodId][0]))
            compareHistograms()            
            printProgress(step,len(OPENCV_METHODS)+len(K_WINNERS))         
    
    os.system('chmod 777 -R ' + arguments.export_dir)   
    print("FINISHED...") 
         

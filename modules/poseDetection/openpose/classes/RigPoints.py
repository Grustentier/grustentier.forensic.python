'''
Created on 07.10.2021

@author: grustentier
'''

class RigPoints: 
        
    @staticmethod
    def getMinX(points): return min([points[i] for i in range(0,len(points),3) if points[i] != 0])
    
    @staticmethod
    def getMaxX(points): return max([points[i] for i in range(0,len(points),3)])   
     
    @staticmethod
    def getMinY(points): return min([points[i] for i in range(1,len(points),3) if points[i] != 0])
    
    @staticmethod
    def getMaxY(points): return max([points[i] for i in range(1,len(points),3)])
    
    @staticmethod    
    def getNose(points): return points[0:3]
    
    @staticmethod
    def getNeck(points): return points[3:6]
    
    @staticmethod
    def getRightShoulder(points): return points[6:9]
    
    @staticmethod
    def getRightElbow(points): return points[9:12]
    
    @staticmethod
    def getRightWrist(points): return points[12:15]
    
    @staticmethod
    def getLeftShoulder(points): return points[15:18]
    
    @staticmethod
    def getLeftElbow(points): return points[18:21]
    
    @staticmethod
    def getLeftWrist(points): return points[21:24]
    
    @staticmethod
    def getHipMiddle(points): return points[24:27]
    
    @staticmethod
    def getRightHip(points): return points[27:30]
    
    @staticmethod
    def getRightKnee(points): return points[30:33]
    
    @staticmethod
    def getRightAnkle(points): return points[33:36]
    
    @staticmethod
    def getLeftHip(points): return points[36:39]
    
    @staticmethod
    def getLeftKnee(points): return points[39:42]
    
    @staticmethod
    def getLeftAnkle(points): return points[42:45]
    
    @staticmethod
    def getRightEye(points): return points[45:48]
    
    @staticmethod
    def getLeftEye(points): return points[48:51]
    
    @staticmethod
    def getRightEar(points): return points[51:54]
    
    @staticmethod
    def getLeftEar(points): return points[54:57]
    
    @staticmethod
    def getLeftBigToe(points):return points[57:60]
    
    @staticmethod
    def getLeftSmallToe(points):return points[60:63]
    
    @staticmethod
    def getLeftHeel(points):return points[63:66]
    
    @staticmethod
    def getRightBigToe(points):return points[63:66]
    
    @staticmethod
    def getRightSmallToe(points):return points[66:69]
    
    @staticmethod
    def getRightHeel(points):return points[69:72]
    
    @staticmethod
    def getBackground(points):return points[72:75]
    
    @staticmethod
    def rigIsComplete(points,rigPointPredictionRatioThreshold=0.6): 
        relevantPoints =  []
        
        ''' more 'static' points '''
        relevantPoints.append(RigPoints.getNeck(points))
        relevantPoints.append(RigPoints.getHipMiddle(points))
        relevantPoints.append(RigPoints.getLeftShoulder(points))
        relevantPoints.append(RigPoints.getRightShoulder(points)) 
        relevantPoints.append(RigPoints.getLeftHip(points))
        relevantPoints.append(RigPoints.getRightHip(points))
        relevantPoints.append(RigPoints.getLeftKnee(points))
        relevantPoints.append(RigPoints.getRightKnee(points))
        
        ''' more moveable points'''
        relevantPoints.append(RigPoints.getLeftElbow(points))
        relevantPoints.append(RigPoints.getRightElbow(points))
        relevantPoints.append(RigPoints.getLeftWrist(points))
        relevantPoints.append(RigPoints.getRightWrist(points))
        
        relevantPoints.append(RigPoints.getLeftAnkle(points))
        relevantPoints.append(RigPoints.getRightAnkle(points))
        
        isComplete = True
    
        for rp in relevantPoints:           
            if 0 in rp:
                isComplete = False
                break
            if rigPointPredictionRatioThreshold is not None and rigPointPredictionRatioThreshold > rp[2]:
                isComplete = False
                break
                    
        return isComplete
    
    @staticmethod
    def getProcessablePoints(completeRigs, position = None):  
    
        marginBottomLeft = 0
        if position is not None:
            marginBottomLeft = int(RigPoints.getLeftAnkle(completeRigs)[1]) - position
        
        marginBottomRight = 0
        if position is not None:
            marginBottomRight = int(RigPoints.getRightAnkle(completeRigs)[1]) - position
        
        points = []
        points.append(["Nose",  [int(RigPoints.getNose(completeRigs)[0]),int(RigPoints.getNose(completeRigs)[1]) - marginBottomLeft,RigPoints.getNose(completeRigs)[2]]])
        points.append(["Neck",  [int(RigPoints.getNeck(completeRigs)[0]),int(RigPoints.getNeck(completeRigs)[1]) - marginBottomLeft,RigPoints.getNeck(completeRigs)[2]]])
        points.append(["l.S.",[int(RigPoints.getLeftShoulder(completeRigs)[0]),int(RigPoints.getLeftShoulder(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftShoulder(completeRigs)[2]]])
        points.append(["r.S.",[int(RigPoints.getRightShoulder(completeRigs)[0]),int(RigPoints.getRightShoulder(completeRigs)[1]) - marginBottomRight,RigPoints.getRightShoulder(completeRigs)[2]]])
        points.append(["l.E.",[int(RigPoints.getLeftElbow(completeRigs)[0]),int(RigPoints.getLeftElbow(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftElbow(completeRigs)[2]]])
        points.append(["r.E.",[int(RigPoints.getRightElbow(completeRigs)[0]),int(RigPoints.getRightElbow(completeRigs)[1]) - marginBottomRight,RigPoints.getRightElbow(completeRigs)[2]]])
        points.append(["l.W.",[int(RigPoints.getLeftWrist(completeRigs)[0]),int(RigPoints.getLeftWrist(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftWrist(completeRigs)[2]]])
        points.append(["r.W.",[int(RigPoints.getRightWrist(completeRigs)[0]),int(RigPoints.getRightWrist(completeRigs)[1]) - marginBottomRight,RigPoints.getRightWrist(completeRigs)[2]]])
        points.append(["l.H.",[int(RigPoints.getLeftHip(completeRigs)[0]),int(RigPoints.getLeftHip(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftHip(completeRigs)[2]]])
        points.append(["r.H.",[int(RigPoints.getRightHip(completeRigs)[0]),int(RigPoints.getRightHip(completeRigs)[1]) - marginBottomRight,RigPoints.getRightHip(completeRigs)[2]]])
        points.append(["l.K.",[int(RigPoints.getLeftKnee(completeRigs)[0]),int(RigPoints.getLeftKnee(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftKnee(completeRigs)[2]]])
        points.append(["r.K.",[int(RigPoints.getRightKnee(completeRigs)[0]),int(RigPoints.getRightKnee(completeRigs)[1]) - marginBottomRight,RigPoints.getRightKnee(completeRigs)[2]]])
        points.append(["l.A.",[int(RigPoints.getLeftAnkle(completeRigs)[0]),int(RigPoints.getLeftAnkle(completeRigs)[1]) - marginBottomLeft,RigPoints.getLeftAnkle(completeRigs)[2]]])
        points.append(["r.A.",[int(RigPoints.getRightAnkle(completeRigs)[0]),int(RigPoints.getRightAnkle(completeRigs)[1]) - marginBottomRight,RigPoints.getRightAnkle(completeRigs)[2]]])
    
        return points
    
    
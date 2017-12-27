# -*- coding:utf-8 -*-

from os import listdir
from numpy import *
import operator
def createDataSet():
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def img2vector(filename):
    # create vector
    returnVect = zeros((1,1024))

    # open the file, and read each line
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        lineInt = map(int, list(lineStr)[0:32])
        returnVect[0, 32*i:(32*(i+1))] = lineInt
        #for j in range(32):
        #    returnVect[0, 32*i+j] = int(lineStr[j])
    
    fr.close()
    return returnVect

def classify0(inX, dataSet, labels, k):
    # get the number of samples
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet

    distances = sum(diffMat**2, axis=1)**0.5
    
    # the argsort() method return the sorted index of each element
    # 第i个位置是第i小的元素原来的index
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def handwritingClassTest(k = 3):
    # labels of sample data
    hwLabels = []

    # lists of sample data files
    trainFileList = listdir('digits/trainingDigits')
    m = len(trainFileList)

    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = trainFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)

        trainingMat[i,:] = img2vector('digits/trainingDigits/%s'%fileNameStr)

    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])

        vectorUnderTest = img2vector('digits/testDigits/%s'%fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print "the classifier came back with: %d, the real answer is: %d"%(classifierResult, classNumStr)

        if classifierResult != classNumStr:
            errorCount += 1.0

    print "\n the total number of errors is: %d"%errorCount
    print "\n the total error rate is: %f"% (errorCount/float(mTest))

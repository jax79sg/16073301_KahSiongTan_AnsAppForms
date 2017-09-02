"""
Labelled data
- appd2v
    - appd2vfilename(Make sure this is sorted), content, label, vector
- cvd2v (limited)
    - cvd2vfilename (Make sure this is sorted), content, label, vector

Unlabelled data
- cvd2v
    - cvd2vfilename (Make sure this is sorted), content, vector
"""
import sys
sys.path.append("/home/kah1/remote_cookie_runtime")
sys.path.append("/home/kah1/remote_cookie_runtime/src")
sys.path.append("/home/kah1/remote_cookie_runtime/src/commons")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/labeller")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromAppJSON")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromCV")
sys.path.append("/home/kah1/remote_cookie_runtime/src/models/mlClassifierModel")
sys.path.append("/home/kah1/remote_cookie_runtime/src/models/doc2vec")
sys.path.append("/home/kah1/remote_cookie_runtime/src/models/Evaluation")
from commons import Utilities

import pandas as pd
import glob
import sys

class SplitFeature():

    util=None
    def __init__(self, logFilename=None, utilObj=None):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)
            self.util.startTimeTrack()

    def getFeaturesAndSplit(self, featuresFolder):
        """
        Will look for cv*.features files in the directory and split every one of them
        :param directory:
        :return:
        """
        for filename in sorted(glob.iglob(featuresFolder + '/cv*.features')):
            self.splitFeatureToPartsBasedOn112record(testSampleFilename=filename)

    def splitFeatureToParts(self,testSampleFilename, stepSize=4, groupSize=36):
        """
        Split a single 144 sized test features into 9 parts. Generally the parts are used for statistical analysis
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20...
        With groupSize of 36, the first element of each group start from 0,0+36,0+36+36,0+36+36+36...
        With stepsize of 4, the first 4 elements are taken, at intervals of 4.
        E.g. 0,1,2,3  36,37,38,39  72,73,74,75  108,109,110,111
        :param testSampleFilename:
        :param stepSize:
        :param groupSize:
        :return: Files will be saved with a XX appended.
        """
        testsetDF=pd.read_csv(testSampleFilename)
        testsetDF=testsetDF.drop_duplicates(['content'])
        testsetDF.reset_index()
        print(testsetDF['label'])
        baseCounter=0
        # stepSize=4
        # groupSize=36
        #totalSize=(testsetDF.shape[0])
        # iterations=4
        overList=[]
        smallList=[]
        counter=1
        for k in range(0,groupSize,stepSize):
            smallList = []
            baseCounter=k
            for i in range(0,stepSize):
                print('Processing index')
                for j in range(baseCounter,baseCounter+stepSize):
                    smallList.append(j)
                    print(j,end=',')
                    #Extract to dataframe
                baseCounter=baseCounter+groupSize
                print('\n')
            overList.append(smallList)
            df=testsetDF.iloc[smallList]
            newfilename=testSampleFilename.split('.')[0]+'.features.'+str(counter).zfill(3)
            print(df['label'])
            df.to_csv(newfilename, ',', mode='w', header=True, index=False,columns=('filename','content','label','vector'))
            counter=counter+1
            # print(df)

        print(overList)
        print(len(overList))
        pass

    def splitFeatureToPartsBasedOn112record(self,testSampleFilename, stepSize=4, groupSize=28):
        """
        Split a single 144(112) sized test features into 9 parts. Generally the parts are used for statistical analysis
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20...
        With groupSize of 36(28), the first element of each group start from 0,0+36(28),0+36(28)+36(28),0+36(28)+36(28)+36(28)...
        With stepsize of 4, the first 4 elements are taken, at intervals of 4.
        E.g. 0,1,2,3  36,37,38,39  72,73,74,75  108,109,110,111  (for 144 records)
        E.g. 0,1,2,3  28,29,30,31  84,85,86,87...    (for 112 records)

        :param testSampleFilename:
        :param stepSize:
        :param groupSize:
        :return: Files will be saved with a XX appended.
        """
        testsetDF=pd.read_csv(testSampleFilename)
        testsetDF=testsetDF.drop_duplicates(['content'])
        testsetDF.reset_index()
        print(testsetDF['label'])
        baseCounter=0
        # stepSize=4
        # groupSize=36
        #totalSize=(testsetDF.shape[0])
        # iterations=4
        overList=[]
        smallList=[]
        counter=1
        for k in range(0,groupSize,stepSize):
            smallList = []
            baseCounter=k
            for i in range(0,stepSize):
                print('Processing index')
                for j in range(baseCounter,baseCounter+stepSize):
                    smallList.append(j)
                    print(j,end=',')
                    #Extract to dataframe
                baseCounter=baseCounter+groupSize
                print('\n')
            overList.append(smallList)
            df=testsetDF.iloc[smallList]
            newfilename=testSampleFilename.split('.')[0]+'.features.'+str(counter).zfill(3)
            print(df['label'])
            df.to_csv(newfilename, ',', mode='w', header=True, index=False,columns=('filename','content','label','vector'))
            counter=counter+1
            # print(df)

        print(overList)
        print(len(overList))
        pass

if(len(sys.argv)==3):
    logLoc = sys.argv[1]
    featuresFolder = sys.argv[2]


    print('logloc: ', logLoc)
    print('featuresFolder: ',featuresFolder)

    splitter=SplitFeature()
    splitter.getFeaturesAndSplit(featuresFolder=featuresFolder)

else:
    print("No arguments provided..")
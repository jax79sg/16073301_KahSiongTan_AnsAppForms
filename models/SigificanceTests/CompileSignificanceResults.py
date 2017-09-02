

import glob
from commons import Utilities
from itertools import combinations
from collections import defaultdict
from itertools import permutations
import pandas as pd

class CompileSignificanceResults():

    testList=[]

    util=None
    TYPE_WILCOXON='wilcoxon'
    TYPE_STUARTMAXWELL='stuart'

    def __init__(self,utilObj=None, logFile=None):
        if (utilObj != None):
            self.util = utilObj
        elif (logFile != None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFile)

    def addTest(self, type=None, folderOfResults=None):
        dictionary=self.collectInit(type=type,folderOfResults=folderOfResults)
        self.testList.append(dictionary)
        self.util.logDebug('CompileSignificanceResults',type + ' significance test added.')

    def collectInit(self, type=None, folderOfResults=None):
        """

        :param type:
        :param folderOfResults:
        :return:
        """
        self.util.logDebug('CompileSignificanceResults','Collecting information for ' + type + ' test')
        dictSignificance=defaultdict(float)
        fileExt=''
        if (type==self.TYPE_STUARTMAXWELL):
            fileExt='smpvalue'
        elif(type==self.TYPE_WILCOXON):
            fileExt='wilpvalue'

        for pvalueResult in (glob.iglob(folderOfResults + '/*.'+fileExt)):
            # -1 means not significant
            # 1 means significant
            # 0 means unable to compute
            result=self.util.readFileContent(pvalueResult)
            finalResult=0
            try:
                resultFloat=float(result)
                finalResult=resultFloat
            except Exception as error:
                finalResult=0
            leftKey, rightKey=pvalueResult.split('/')[-1].split('.')[0].split('-')
            key1=leftKey.lower()+'-'+rightKey.lower()
            key2 = rightKey.lower() + '-' + leftKey.lower()
            dictSignificance[key1.lower()]=finalResult
            dictSignificance[key2.lower()] = finalResult
        return dictSignificance

    def createSigTable(self,listOfApproaches, saveFilename=None):
        """
        First added test has priority
        :param listOfApproaches: List of strings representing the methods. A pairwise comparison will be returned.
        :return:
        """
        listOfApproaches=[x.lower() for x in listOfApproaches]
        if len(self.testList)==0:
            self.util.logError('CompileSignificanceResults','You need to addTest() before creating a sigtable...exiting')
            exit(-1)
        updatedSignificanceDF=pd.DataFrame(0.0, index=listOfApproaches, columns=listOfApproaches)
        for dictSignificance in self.testList:
            df = pd.DataFrame(0.0, index=listOfApproaches, columns=listOfApproaches)
            # print('Empty Frame:',df)
            # print(df.at['one','two'])
            # df.at['two', 'two']=-1
            # print('2x2 gotsomething?:', df)
            combo = combinations(listOfApproaches, 2)
            for item in combo:
                #For each combination,get the result and put into the dataframe table.

                #Getting results.
                left= item[0].lower()
                right = item[1].lower()
                pvalue=dictSignificance[left+'-'+right]
                df.at[left,right]=pvalue
                df.at[right, left] = pvalue
            updatedSignificanceDF=updatedSignificanceDF+df
            self.util.logInfo('CompileSignificanceResults','Significance Table:\n'+df.to_string())

        self.util.logInfo('CompileSignificanceResults','Final table:\n'+updatedSignificanceDF.to_string())
        updatedSignificanceDF.to_csv(saveFilename, ',', mode='w',header=True, index=True, columns=listOfApproaches)
        return updatedSignificanceDF


if __name__ == "__main__":
    util=Utilities.Utility()
    util.setupLogFileLoc('/home/kah1/CompileSigResults.log')
    sigTest=CompileSignificanceResults(utilObj=util)
    sigTest.addTest(type=CompileSignificanceResults.TYPE_WILCOXON,folderOfResults='/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/significance/wilcoxon')
    sigTest.addTest(type=CompileSignificanceResults.TYPE_STUARTMAXWELL,folderOfResults='/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/significance/stuart')
    sigTest.createSigTable(listOfApproaches=['MLP_BOW5000','MLP_D2V100','MLP_TFIDF5000','MLP_W2V100min1','NB_BOW5000','NBG_D2V100','NBG_W2V100min1','NB_TFIDF5000','SIM_BOW5000','SIM_D2V100','SIM_TFIDF5000','SIM_W2V100min1','SVM_BOW5000','SVM_D2V100','SVM_TFIDF5000','SVM_W2V100min1','TOPIC_NONE'])
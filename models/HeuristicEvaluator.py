
"""
Heuristic evaluator do not have any labelled CV sections to score against.
Instead,for every predicted class 'Edu','Skills','Personal Details' and 'Work Experience',
It will pull the relevant Application Form's content for that class. (E.g. From /u01/bigdata/01b_d2v/032/edu/doc2vecEdu)
    - Each CV Section already have the relevant Application Form ID encoded in the filename.
A word by word comparison is made between content of Application Form Predicted class content and the CV section.
If there's sufficient matches, then considered a HIT. Else a miss.
The nature of this Evaluator means it can only provide the Accuracy Metric as the performance metric.
The motivation of this Evaluator
- Not feasible to have large number of labelled CVs
- Will use this Evaluator for approximate evaluation on larger scale.
"""

import sys
sys.path.append("/home/kah1/remote_gitlab_source")
sys.path.append("/home/kah1/remote_gitlab_source/python")
sys.path.append("/home/kah1/remote_gitlab_source/python/extractFromAppJSON")
sys.path.append("/home/kah1/remote_gitlab_source/python/extractFromCV")
import matplotlib

matplotlib.use('Agg')

from commons import Utilities
from collections import defaultdict
import glob
import pandas as pd
import os
from data.extractFromAppJSON import AppJsonReader



class HeuristicEvaluator():

    SCORE_ACCURACY='acc'
    SCORE_CONFUSIONMATRIX='confmatrix'
    SCORE_CLASSREPORT='class'
    SCORE_F1='f1'
    SCORE_PRECISION='precision'
    SCORE_RECALL='recall'
    SCORE_F1_PERCLASS='f1_perclass'
    SCORE_PRECISION_PERCLASS='precision_perclass'
    SCORE_RECALL_PERCLASS='recall_perclass'

    listOfAppD2vFoldersOrderByLabel=None
    listOfLabelsOrderByLabel=None
    appFolder=None

    __logFile=''
    util=Utilities.Utility()

    def __init__(self, logFile=None, utilObj=None, listOfAppD2vFoldersOrderByLabel=None, listOfLabelsOrderByLabel=None, appFolderLocation=None):
        if (utilObj != None):
            self.util = utilObj
        elif (logFile != None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFile)
        if (listOfAppD2vFoldersOrderByLabel==None or listOfLabelsOrderByLabel==None):
            self.util.logError('HeuristicEvaluator','listOfAppD2vFoldersOrderByLabel and their corresponding listOfLabelsOrderByLabel needs to be provided...cannot proceed')
            exit(-1)
        self.listOfLabelsOrderByLabel=listOfLabelsOrderByLabel
        self.listOfAppD2vFoldersOrderByLabel=listOfAppD2vFoldersOrderByLabel
        self.appFolder=appFolderLocation

    def generateGraphicalConfusionMatrix(self, array):
        self.util.logInfo('HeuristicEvaluator','Confusion matrix is not available for Heuristic Evaluation')

    def generateSummary(self, folderpath):
        from decimal import Decimal
        metricDict=defaultdict(list)
        for metricFile in sorted(glob.iglob(folderpath + '/*.metric')):
            tokens=metricFile.split('/')[-1].split('_')
            approach=tokens[0]
            vsm=tokens[1]
            metric=tokens[2].split('.')[0]
            value=str(round(Decimal(self.util.readFileContent(metricFile)),2))
            recordTuple=(approach,vsm,value)
            if(metric in metricDict):
                tupleList=metricDict[metric]
                tupleList.append(recordTuple)
                metricDict[metric]=tupleList
            else:
                tupleList=[recordTuple]
                metricDict[metric] = tupleList

        metricTuple=[]
        for key,value in  metricDict.items():

            metricResultsDF = pd.DataFrame(columns=(['VSM']))
            metricResultsDF = metricResultsDF.set_index("VSM")
            for tupleList in value:
                approach=tupleList[0]
                vsm = tupleList[1]
                metricValue = tupleList[2]
                metricResultsDF.set_value(vsm, approach, metricValue)
            # print(metricResultsDF)
            metricResultsDF.to_csv(folderpath+'/'+key+'.summary',header=True,index_label='VSM',index=True,mode='w')




    def printSummary(self):
        self.util.logInfo('HeuristicEvaluator','printSummary is not implemented')

    def score(self,y=None, ypred=None,type=SCORE_ACCURACY, filename=None, **kwargs):
        heuristicResults = pd.DataFrame(columns=('ACTUAL', 'PREDICTED', 'HEURISTIC_HIT', 'HIT_SCORE','HIT_REASONS','APPCONTENT','CVCONTENT'))
        testDF=None
        if(y!=None):
            self.util.logInfo('HeuristicEvaluator','the y variable will not be used as Heuristic Evaluator do not rely on labels.')
        for key,value in kwargs.items():
            if(key=='testDF'):
                testDF=value
        results=''
        if(type==self.SCORE_ACCURACY):

            # Content of testDF: cvd2vfilename, content, label, vector
            counter=0
            hitCounter=0
            for index,item in testDF.iterrows():
                pass
                currentYpred=ypred[counter]
                actualLabel=y[counter]
                labelPosition=self.listOfLabelsOrderByLabel.index(currentYpred)

                labelNames=self.listOfAppD2vFoldersOrderByLabel[labelPosition]


                cvSectionContent=item['content']
                cvTokens=self.util.tokenize(cvSectionContent)
                cvD2vFilename=item['filename']
                clientid, appid, hashcode, paraid=self.util.getInfoFromCVd2vFilename(cvD2Vfilename=cvD2vFilename)

                folderOfAppD2VForLabel=self.util.pathfinder(type=self.util.TYPE_APP,filename='app_'+str(clientid).zfill(3)+'_'+str(appid)+'.json',prefix=self.appFolder)

                appJsonReader=AppJsonReader.AppJsonReader(folderOfAppD2VForLabel)
                labelsDF=appJsonReader.getAllPageTitlesAndAnswers()
                appContent=''
                appD2vFound = False
                for index, item in labelsDF.iterrows():
                    # 'label', 'aquestion', 'aanswer'

                    label=item['label']
                    # print('Processing ' + label)

                    for labelName in labelNames.split(':'):
                        if(labelName.lower() in label.lower()):
                            appD2vFound = True
                            appContent=appContent+' '+item['aanswer'] + ' '
                        pass
                appContentTokens = self.util.tokenize(appContent)

                if(appD2vFound==False):
                    self.util.logError('Heuristic','AppID ' + str(folderOfAppD2VForLabel)+ ' for prediction of ' + ','.join(labelNames)  + ' not found.' + '\nActual label is ' + str(actualLabel))

                identicals = set(appContentTokens) & set(cvTokens)
                # self.util.logDebug('HeuristicEvaluation','appContentTokens: ' + ' '.join(appContentTokens))
                # self.util.logDebug('HeuristicEvaluation', 'cvTokens: ' + ' '.join(cvTokens))
                # self.util.logDebug('HeuristicEvaluation', 'identicals: ' + ' '.join(identicals))
                newEntry=None

                cleansedIdenticals=[]
                for identical in identicals:
                    import re
                    word = identical
                    regexp = re.compile(r'(\b\d{1}\b)|(\b\d{2}\b)|(\b\d{3}\b)')
                    if regexp.search(word):
                        pass
                    else:
                        cleansedIdenticals.append(identical)

                identicals=cleansedIdenticals

                if (len(identicals)>1):
                    hitCounter=hitCounter+1
                    newEntry = pd.DataFrame(
                        data={'ACTUAL': [actualLabel], 'PREDICTED': [currentYpred], 'HEURISTIC_HIT': [1],
                              'HIT_SCORE': [len(identicals)], 'HIT_REASONS': [','.join(identicals)], 'APPCONTENT': [','.join(appContentTokens)], 'CVCONTENT': [','.join(cvTokens)]})
                else:
                    if(appD2vFound==True):
                        newEntry = pd.DataFrame(
                            data={'ACTUAL': [actualLabel], 'PREDICTED': [currentYpred], 'HEURISTIC_HIT': [0],
                                  'HIT_SCORE': [len(identicals)]})
                    else:
                        newEntry = pd.DataFrame(
                            data={'ACTUAL': [actualLabel], 'PREDICTED': [currentYpred], 'HEURISTIC_HIT': [-1],
                                  'HIT_SCORE': [len(identicals)]})
                heuristicResults = heuristicResults.append(newEntry)
                counter=counter+1
            accuracy=hitCounter/counter
            # self.util.logInfo('HeuristicEvaluator','Results of the Heuristic Evaluation')
            pd.set_option("display.max_colwidth", 10000)
            # self.util.logInfo('HeuristicEvaluator',heuristicResults.to_string(columns=['ACTUAL', 'PREDICTED', 'HEURISTIC_HIT', 'HIT_SCORE', 'HIT_REASONS','APPCONTENT','CVCONTENT']))
            # print(y)
            # print(ypred)
            results=accuracy
            self.util.saveStringToFile(results,filename=filename+'_HEUACC.metric')

        elif(type==self.SCORE_CONFUSIONMATRIX):
            self.util.logInfo('HeuristicEvaluator', 'Confusion matrix is not available for Heuristic Evaluation')
        elif(type==self.SCORE_CLASSREPORT):
            self.util.logInfo('HeuristicEvaluator', 'Classification report is not available for Heuristic Evaluation')
        elif(type==self.SCORE_F1_PERCLASS):
            self.util.logInfo('HeuristicEvaluator', 'F1 is not available for Heuristic Evaluation')
        elif(type==self.SCORE_PRECISION_PERCLASS):
            self.util.logInfo('HeuristicEvaluator', 'Precision is not available for Heuristic Evaluation')
        elif(type==self.SCORE_RECALL_PERCLASS):
            self.util.logInfo('HeuristicEvaluator', 'Recall is not available for Heuristic Evaluation')
        elif(type==self.SCORE_F1):
            self.util.logInfo('HeuristicEvaluator', 'F1 is not available for Heuristic Evaluation')
        elif(type==self.SCORE_PRECISION):
            self.util.logInfo('HeuristicEvaluator', 'Precision is not available for Heuristic Evaluation')
        elif(type==self.SCORE_RECALL):
            self.util.logInfo('HeuristicEvaluator', 'Recall is not available for Heuristic Evaluation')
        return  results


    def _getFirstString(self,myDelimitedStr=None, delimiter=','):
        """
        Return the first string from a set of delimited strings
        :param categoryStr:
        :param delimiter:
        :return:
        """
        try:
            results=myDelimitedStr.split(delimiter)[0]
        except Exception as error:
            results=''
        return (results)


    def evaluateHeuristic(self, resultsDatasetFilename=None, heuristicsFilename=None, appd2vEduFolder=None, appd2vSkillsFolder=None, appd2vPersonalDetailsFolder=None):
        """
         resultsDataset contains the categorised results from Test phase 2.
        This will have information on appid, clientid, cvd2vfilename,and its inferred categories (edu, skills, personaldetails) and corresponding scores.
        appd2vEduFolder,appd2vSkillsFolder and appd2vPersonalDetailsFolder will contain the appd2v files which are named app_xx_yyyy.d2v.
        For each row in resultsDataset,
        - pull the content of cvd2vfilename.
        - pull the content of appd2v[category]Folder/app_xx_yyyy.d2v.
        - Strip the stop words
        compare every word in both content,
        - if the number of words identical hit a certain threshold, then HIT.
        - else, MISS
        Add this HIT/MISS into the resultsDataset csv as a new column.
        Add identical words as a new column
        :param resultsDatasetFilename: Names of files seperated by ';'
        :param heuristicsFilename: Name of the file to save results in
        :param appd2vEduFolder: Folder that contain the appd2v files which are named app_xx_yyyy.d2v.
        :param appd2vSkillsFolder: Folder that contain the appd2v files which are named app_xx_yyyy.d2v.
        :param appd2vPersonalDetailsFolder: Folder that contain the appd2v files which are named app_xx_yyyy.d2v.
        :return:
        """
        filecounter=0
        resultsDatasetFilenames=resultsDatasetFilename.split(';')
        fullResults = pd.DataFrame(columns=(
            'appid', 'clientid', 'cvd2vfilename', 'categories', 'scores', 'heuristics', 'heuristics_reason','content'))

        hitCounter=0
        MissCounter = 0
        for resultsDatasetFilename in resultsDatasetFilenames:
            #To confirm if the filename contains headers.
            self.util.logDebug('Evaluator-evaluateFromDataset', 'Reading ' + resultsDatasetFilename)
            resultsDF=None
            resultsDF=pd.read_csv(resultsDatasetFilename, header=None)

            #Should load as 'appid', 'clientid', 'cvd2vfilename','content', 'categories', 'scores'
            self.util.logDebug('Evaluator-evaluateFromDataset', 'Processing...')

            counter=0
            errcounter=0

            for index, row in resultsDF.iterrows():
                # try:
                    clientid=row[1]
                    clientid=(str(clientid)).zfill(3)
                    appid=row[0]
                    cvd2vFullpath=row[2]
                    categories=row[4]
                    scores=row[5]
                    content=row[3]
                    #The heuristics will only take the category with highest score.
                    category=self._getFirstString(myDelimitedStr=categories, delimiter=':')
                    score=self._getFirstString(myDelimitedStr=scores, delimiter=':')

                    cvd2vContent = open(cvd2vFullpath, 'r').read()
                    cvd2vContentTokens=self.util.tokenize(cvd2vContent)   #This is to be used for comparison.
                    heuristics_reason=''
                    appd2vFilename=''
                    #Based on the matched category, pull the relevant app_xx_yyyy.d2v file.
                    if (category==self.util.LOOKUP_CAT_EDU):
                        appd2vFilename=appd2vEduFolder+'/'+'app_'+str(clientid)+'_'+str(appid)+'.'+ self.util.LOOKUP_EXT_APPD2V

                    elif(category==self.util.LOOKUP_CAT_SKILLS):
                        appd2vFilename = appd2vSkillsFolder + '/' + 'app_' + str(clientid) + '_' + str(
                            appid) + '.' + self.util.LOOKUP_EXT_APPD2V

                    elif (category == self.util.LOOKUP_CAT_PERSONALDETAILS):
                        appd2vFilename = appd2vPersonalDetailsFolder + '/' + 'app_' + str(clientid) + '_' + str(
                            appid) + '.' + self.util.LOOKUP_EXT_APPD2V

                    if (os.path.exists(appd2vFilename)==True):
                        appd2vContent = open(appd2vFilename, 'r').read()
                    else:
                        appd2vContent=''
                        heuristics_reason='FILE_NOT_FOUND_IN_CATEGORY: '+appd2vFilename
                    appd2vContentTokens = self.util.tokenize(appd2vContent)  # This is to be used for comparison.

                    identicals=set(appd2vContentTokens) & set(cvd2vContentTokens)
                    # if(len(identicals)>0):
                    #     heuristics_reason=self.util.tokensToStr(identicals)
                    heuristics='MISS'
                    if(category==self.util.LOOKUP_CAT_EDU and len(identicals)>=self.util.THRES_EDU):
                        heuristics='HIT'
                        heuristics_reason = self.util.tokensToStr(identicals)
                    elif(category==self.util.LOOKUP_CAT_SKILLS and len(identicals)>=self.util.THRES_SKILLS):
                        heuristics = 'HIT'
                        heuristics_reason = self.util.tokensToStr(identicals)
                    elif (category == self.util.LOOKUP_CAT_PERSONALDETAILS and len(identicals) >= self.util.THRES_PERSONALDETAILS):
                        ## For personal details, can be more restrictive by limiting to words not in English Dictionary
                        identicals=self.util.returnNonEnglishDictWords(identicals)
                        if(len(identicals)>self.util.THRES_PERSONALDETAILS):
                            heuristics = 'HIT'
                            heuristics_reason = self.util.tokensToStr(identicals)

                    # print(heuristics)
                    currentRow = pd.DataFrame(data={'appid': [appid], 'clientid': [clientid],'cvd2vfilename': [resultsDatasetFilename], 'categories': [categories],'scores': [scores], 'heuristics': [heuristics],'heuristics_reason': [heuristics_reason],'content': [cvd2vContent]})
                    fullResults=fullResults.append(currentRow)
                    counter = counter + 1
                    if counter%100 ==0:
                        self.util.logDebug('Evaluator-evaluateFromDataset',
                                             str(counter) + ' files completed with ' + str(errcounter) + ' errors.')
                        self.util.logDebug('Evaluator-evaluateFromDataset', 'Saving!')
                        fullResults.to_csv(heuristicsFilename.split('.')[0]+'_'+str((filecounter)).zfill(2)+'.csv', ',', mode='a', header=False, index=False,
                                           columns=['appid', 'clientid', 'cvd2vfilename', 'categories', 'scores', 'heuristics', 'heuristics_reason', 'content'])
                        fullResults = fullResults[0:0]
                # except Exception as error:
                #     errcounter=errcounter+1
                #     self.util.logDebug('Evaluator-evaluateFromDataset', 'Error encountered: ' + repr(error))
            self.util.logDebug('Evaluator-evaluateFromDataset', 'Final save!')
            fullResults.to_csv(heuristicsFilename.split('.')[0]+'_'+str((filecounter)).zfill(2)+'.csv', ',', mode='a', header=False, index=False,
                               columns=['appid', 'clientid', 'cvd2vfilename', 'categories', 'scores', 'heuristics',
                                        'heuristics_reason', 'content'])
            filecounter=filecounter+1

# # TestA phase 1xa
# # python3 Evaluator.py '/u01/bigdata/03a_01b_test/cvd2v/test2/heuEval.log' '/u01/bigdata/03a_01b_test/cvd2v/test2/TestA_32_userlabelled_results_1000.csv;/u01/bigdata/03a_01b_test/cvd2v/test2/TestA_32_userlabelled_results_2000.csv;/u01/bigdata/03a_01b_test/cvd2v/test2/TestA_32_userlabelled_results_2759.csv' '/u01/bigdata/03a_01b_test/cvd2v/test2/heuEval.csv' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu' '/u01/bigdata/01b_d2v/032/skills/doc2vecSkills' '/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails'
# if __name__ == "__main__":
#     if(len(sys.argv)==7):
#         logFile = sys.argv[1]
#         resultsDatasetFilename = sys.argv[2]
#         heuristicsFilename=(sys.argv[3])
#         appd2vEduFolder=(sys.argv[4])
#         appd2vSkillsFolder=sys.argv[5]
#         appd2vPersonalDetailsFolder=sys.argv[6]
#         print('Logging to ', logFile)
#         print('resultsDatasetFilename',resultsDatasetFilename)
#         print('heuristicsFilename', heuristicsFilename)
#         print('appd2vEduFolder', appd2vEduFolder)
#         print('appd2vSkillsFolder', appd2vSkillsFolder)
#         print('appd2vPersonalDetailsFolder', appd2vPersonalDetailsFolder)
#
#         eval=Evaluator(logFile)
#         eval.evaluateHeuristic(resultsDatasetFilename=resultsDatasetFilename, heuristicsFilename=heuristicsFilename, appd2vEduFolder=appd2vEduFolder, appd2vSkillsFolder=appd2vSkillsFolder, appd2vPersonalDetailsFolder=appd2vPersonalDetailsFolder)
#     else:
#         print('Arguments incorrect')

# e=Evaluator()
# e.generateSummary('/u01/bigdata/02d_d2vModel1/features')
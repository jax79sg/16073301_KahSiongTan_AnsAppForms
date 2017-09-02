"""
Inverse Answering by Topic

Input: labelled CV test examples
Input: Pre-trained LDA model and assigned topic
Output: predicted CV test examples
Pipe: Pipe predicted CV test examples and labelled CV test examples to Evaluator class.

For each labelled CV test example
    - Extract filename, content
    - Predict topic from content
    - Save filename, content and predicted topic (May need to rearrange the topic number)

Pass saved predicted results and labelled test examples into evaluator class for evaluation.
"""

from commons import Utilities
from features import LDA
import pandas as pd
def dataInsert(index, item):
    content = item['content']
    label = int(item['label'])
    # sampleDim = sampleDim + 1
    # xDim = len(vector)
    return (label, content)

class TopicClusteringPredictionModel():
    _model=None
    _util=None
    _Xtest=[]
    _Ytest=[]
    _YPredTest=[]
    _trained=False
    testsetDF=None

    def __init__(self, logFile=None, utilObj=None, ldaModelFilename=None):
        """

        :param ldaModel:
        :param topicMapping:
        :param labelledTestSamples: Can be any features since the vector doesn't matter.
        """
        if (utilObj != None):
            self._util = utilObj
        elif (logFile != None):
            self._util = Utilities.Utility()
            self._util.setupLogFileLoc(logFile)

        self._model=LDA.LDA(utilObj=self._util)
        if (ldaModelFilename!=None):
            self._model.loadModel(ldaModelFilename)

        _trained=True
        pass

    def getXtest(self):
        return self._Xtest

    def getYtest(self):
        return self._Ytest

    def flushTestData(self):
        self._Xtest = []
        self._Ytest = []
        self._YPredTest = []
        self.testsetDF = None

    def savePredictions(self, filename=None):
        """
        Save the label and prediction into a file of format
        label,prediction
        :return:
        """
        stringToSave='label,prediction\n'
        for index in range(0,len(self._YPredTest)):
            label=self._Ytest[index]
            predict=self._YPredTest[index]
            stringToSave=stringToSave+str(label)+','+str(predict)+'\n'
        self._util.saveStringToFile(stringToSave,filename=filename+'.predictions')

    def loadXYtest(self, testSampleFilename=None):
        self._util.logDebug('TopicClusteringPredictionModel','Loading test set')
        self._util.logDebug('TopicClusteringPredictionModel', 'Reading CSV ' + testSampleFilename)
        testsetDF=pd.read_csv(testSampleFilename)
        self._util.logDebug('TopicClusteringPredictionModel', 'Read CSV in ' + self._util.checkpointTimeTrack())
        testsetDF=testsetDF.drop_duplicates(['content'])
        self.testsetDF=testsetDF
        multicore = False
        if (multicore == True):
            self._util.logDebug('TopicClusteringPredictionModel', 'Processing into list (Multicore)...')
            from joblib import Parallel, delayed
            myset = Parallel(n_jobs=5)(delayed(dataInsert)(index, item) for index, item in testsetDF.iterrows())
            label, content = zip(*myset)
            self._Xtest = content
            self._Ytest = label
        else:
            self._util.logDebug('MLClassifierPredictionModel', 'Processing into list (Single core)...')
            for index, item in testsetDF.iterrows():
                content = item['content']
                label = int(item['label'])
                self._Xtest.append(content)
                self._Ytest.append(label)
                # sampleDim = sampleDim + 1








        # for index,item in testsetDF.iterrows():
        #     content=item['content']
        #     label=int(item['label'])
        #     self._Xtest.append(content)
        #     self._Ytest.append(label)
        self._util.logDebug('TopicClusteringPredictionModel', 'Test set loaded in ' + self._util.checkpointTimeTrack())

    def train(self, trainingFilename=None):
        self._util.logInfo('TopicClusteringPredictionModel','This model comes pretrained. If you are looking at reclustering the topics, refer to features.LDA.trainModel')

    def predict_proba(self, x):
        import numpy as np
        predictionList=[]
        if self._model.loaded:
            for xValue in x:
                correctedProba=[0.0,0.0,0.0,0.0]
                predictProba=self._model.infer_topic_proba(xValue)

                index=0
                for proba in predictProba:
                    realClassID=int(self._model.topicLabelling[index])
                    correctedProba[realClassID]=proba
                    index=index+1
                predictionList.append(correctedProba)
        else:
            self._util.logError('TopicClusteringPredictionModel','Model needs to be loaded before prediction')

        predictionList=np.array(predictionList)
        return predictionList


    def predict(self, x):
        """
        Predict a topic id based on x content. The topic id is mapped to a user topic id before returning.
        :param x:
        :return:An integer refering to a topic as defined by user.
        """
        predictionList=[]
        if self._model.loaded:
            for xValue in x:
                systemLabel=self._model.infer_topic(xValue)
                result=self._model.topicLabelling[systemLabel]
                predictionList.append(int(result))
        else:
            self._util.logError('TopicClusteringPredictionModel','Model needs to be loaded before prediction')

        return predictionList

    def evaluate(self, approach_vsm_filename=None, eval=None):
        self._YPredTest=[]
        finalResults=[]
        if (len(self._Xtest)==len(self._Ytest) and len(self._Xtest)>0 or len(self._Ytest)>0 and self._trained==True):
            self._util.logDebug('TopicClusteringPredictionModel', 'Predicting Y for all X')
            # for x in self._Xtest:
            self._YPredTest=self.predict(self._Xtest)
            # test=self.predict_proba(self._Xtest)
                # self._YPredTest.append(int(yPred))
            self._util.logDebug('TopicClusteringPredictionModel', 'Predicting Y for all X completed')
            # eval=Evaluator.Evaluator(utilObj=self._util)
            self.savePredictions(filename=approach_vsm_filename)
            resultsAcc = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_ACCURACY, filename=approach_vsm_filename, testDF=self.testsetDF)
            self._util.logInfo('TopicClusteringPredictionModel','Accuracy is ' + str(resultsAcc))

            resultsPrec = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_PRECISION, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','Precision is ' + str(resultsPrec))

            resultsRecall = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_RECALL, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','Recall is ' + str(resultsRecall))

            resultsF1 = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_F1, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','F1 is ' + str(resultsF1))


            resultsF1perclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_F1_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','F1 is ' + str(resultsF1perclass))

            resultsPrecperclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_PRECISION_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','Prec is ' + str(resultsPrecperclass))

            resultsRecallperclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_RECALL_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','F1 is ' + str(resultsRecallperclass))

            resultsClass=eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_CLASSREPORT, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','Classification is \n' + resultsClass)

            resultsConfu=eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_CONFUSIONMATRIX, filename=approach_vsm_filename)
            self._util.logInfo('TopicClusteringPredictionModel','Confusion is \n' + resultsConfu)

            finalResults = resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass
        else:
            self._util.logError('TopicClusteringPredictionModel', 'X and Y needs to be loaded before prediction!')

        return finalResults
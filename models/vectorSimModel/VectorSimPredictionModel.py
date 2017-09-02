"""

Inverse Answering by Similarity

Input: Labelled training data (app:content, vector, label)
        - Remove content dups
Input: Labelled test data (cv: content, vector, label)
        - Remove content dups

Create similarity matrix
- X as training data
- Y as test data
        cv1     cv2     cv3 ... cN
app1
app2
app3
.
.
appN

For each column compute TopN similar apps.
From here see if can relate to certain labels or not
"""


from models.vectorSimModel.Similarity import Similarity
from commons import Utilities
import numpy as np
import pandas as pd

def dataInsert(index, item):
    label = int(item['label'])
    vector = list(eval(item['vector']))
    # sampleDim = sampleDim + 1
    # xDim = len(vector)
    return (label, vector)

class VectorSimPredictionModel():
    _model=None
    _util=None
    _trainsetDF=None
    _testsetDF=None
    _Xtest=[]
    _Ytest=[]
    _Xtrain=[]
    _Ytrain=[]

    _YPredTest=[]
    _trained=False
    testsetDF = None

    def __init__(self, logFile=None, utilObj=None):
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
        self._util.startTimeTrack()

    def flushTestData(self):
        self._testsetDF = None
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
        self._util.logDebug('VectorSimPredictionModel','Loading test set')
        self._util.logDebug('VectorSimPredictionModel', 'Reading CSV')
        self._testsetDF=pd.read_csv(testSampleFilename)
        self._util.logDebug('VectorSimPredictionModel', 'Read CSV in ' + self._util.checkpointTimeTrack())
        self._testsetDF=self._testsetDF.drop_duplicates(['content'])
        self.testsetDF = self._testsetDF
        self._util.logDebug('VectorSimPredictionModel', 'Processing into list...')
        for index,item in self._testsetDF.iterrows():
            vector = list(eval(item['vector']))
            label=int(item['label'])
            self._Xtest.append(vector)
            self._Ytest.append(label)
        self._util.logDebug('VectorSimPredictionModel', 'Test set loaded in ' + self._util.checkpointTimeTrack())

    def train(self, trainingFilename=None):

        self._util.logInfo('VectorSimPredictionModel','This model doesn\'t require training but will use the training data for similarity computation.')
        self._util.logDebug('VectorSimPredictionModel','Building labelled training for mapping')
        try:
            self._util.logDebug('VectorSimPredictionModel', 'Reading CSV')
            self._trainsetDF=pd.read_csv(trainingFilename, engine='python')
        except Exception as error:
            self._util.logError('VectorSimPredictionModel',
                                'Error loading the training set,likely too big. Switching to Python engine which is stable but slower')
            self._trainsetDF = pd.read_csv(trainingFilename, engine='python')
            self._util.logDebug('VectorSimPredictionModel', 'Read CSV in ' + self._util.checkpointTimeTrack())

        self._trainsetDF=self._trainsetDF.drop_duplicates(['content'])
        sampleDim=0
        self._trainsetDF=self._trainsetDF.reset_index()



        multicore = True
        if (multicore == True):
            self._util.logDebug('VectorSimPredictionModel', 'Processing into list (Multicore)...')
            from joblib import Parallel, delayed
            myset = Parallel(n_jobs=5)(delayed(dataInsert)(index, item) for index, item in self._trainsetDF.iterrows())
            myLabel, myVector = zip(*myset)
            self._Xtrain = myVector
            self._Ytrain = myLabel
        else:
            self._util.logDebug('VectorSimPredictionModel', 'Processing into list (Single core)...')
            for index, item in self._trainsetDF.iterrows():
                label = int(item['label'])
                vector = list(eval(item['vector']))
                self._Xtrain.append(vector)
                self._Ytrain.append(label)
                # sampleDim = sampleDim + 1
                # xDim = len(vector)

        # self._util.logDebug('VectorSimPredictionModel', 'Processing into list...')
        # for index, item in self._trainsetDF.iterrows():
        #     label = int(item['label'])
        #     vector = list(eval(item['vector']))
        #     self._Xtrain.append(vector)
        #     self._Ytrain.append(label)
        #     sampleDim = sampleDim + 1
        self._util.logInfo('VectorSimPredictionModel',
                          'Training data loaded in ' + self._util.checkpointTimeTrack())
        self._trained=True
        # print(self._trainsetDF.shape)
        # print(self._trainsetDF.loc[[8639]])

    def getXtest(self):
        return self._Xtest

    def getYtest(self):
        return self._Ytest

    def predict_proba(self,x):
        """
        Returns probability instead of actual winner labels
        :param x:
        :return:
        """
        results = []
        sim = Similarity(utilObj=self._util)
        similarityMatrix = sim.cosine_sim(self._Xtrain, x)
        self._util.logDebug('VectorSimPredictionModel',
                            'Similarity matrix generated in ' + self._util.checkpointTimeTrack())

        # print('Check')
        # print(self._trainsetDF.iloc[2])
        # print(self._Xtrain[2])
        # print('Check')
        # print(self._trainsetDF.iloc[200])
        # print(self._Xtrain[200])
        # print('Check')
        # print(self._testsetDF.iloc[200])
        # print(self._Xtest[200])
        # print('Check')
        # print(self._testsetDF.iloc[2])
        # print(self._Xtest[2])



        # print('Sim matrix')
        # print(similarityMatrix)

        # Transpose it into column for easier computation
        trans_similarityMatrix = similarityMatrix.transpose()

        # print('Transpose Sim matrix')
        # print(trans_similarityMatrix)

        # Row is the test data
        # Col is the train data
        # Cell is the similarity

        similarityMatrixDF = pd.DataFrame(trans_similarityMatrix)

        for index, item in similarityMatrixDF.iterrows():
            # print('Unsorted item')
            # print(item)

            # Sort descending, e.g. Highest similarity first (index position is retained)
            sortedItem = item.sort_values(ascending=False, kind='mergesort')
            # print('Sorted item')
            # print(sortedItem)
            # print('Top 5')

            # Get the top five items
            top5item = sortedItem[:5]
            # print(top5item)
            # print(type(top5item))

            # Each element in array represent class Edu, Skills, Personal Details, Work Experience and None.
            totalScore = np.array([0, 0, 0, 0, 0])
            for index, item in top5item.iteritems():
                # Get index info so can check what it represents.
                trainrecord = (self._trainsetDF.loc[[index]])
                # print('Index:'+str(index)+ '  item:'+ str(item))
                if (int(trainrecord['label']) == 0):
                    currentScore = [1, 0, 0, 0, 0]
                elif (int(trainrecord['label']) == 1):
                    currentScore = [0, 1, 0, 0, 0]
                elif (int(trainrecord['label']) == 2):
                    currentScore = [0, 0, 1, 0, 0]
                elif (int(trainrecord['label']) == 3):
                    currentScore = [0, 0, 0, 1, 0]
                else:
                    currentScore = [0, 0, 0, 0, 1]
                tempScore = np.vstack((totalScore, currentScore))
                totalScore = tempScore.sum(axis=0)

            # Divide each vote with the sum.
            totalVotes=totalScore.sum()
            probScore=[x/totalVotes for x in totalScore]
            results.append(probScore)
            # SECTION END
        results=np.array(results)
        results = np.delete(results, -1, axis=1)
        return results
        pass

    def predict(self, x):
        """
        Prediction is based on similarity approach.
        """
        # SECTION START
        # Generate a similarity matrix between the vectors in training and the unseen test vectors
        results=[]
        sim = Similarity()
        similarityMatrix = sim.cosine_sim(self._Xtrain, x)
        self._util.logDebug('VectorSimPredictionModel',
                            'Similarity matrix generated in ' + self._util.checkpointTimeTrack())

        # print('Check')
        # print(self._trainsetDF.iloc[2])
        # print(self._Xtrain[2])
        # print('Check')
        # print(self._trainsetDF.iloc[200])
        # print(self._Xtrain[200])
        # print('Check')
        # print(self._testsetDF.iloc[200])
        # print(self._Xtest[200])
        # print('Check')
        # print(self._testsetDF.iloc[2])
        # print(self._Xtest[2])



        # print('Sim matrix')
        # print(similarityMatrix)

        # Transpose it into column for easier computation
        trans_similarityMatrix = similarityMatrix.transpose()

        # print('Transpose Sim matrix')
        # print(trans_similarityMatrix)

        # Row is the test data
        # Col is the train data
        # Cell is the similarity

        similarityMatrixDF = pd.DataFrame(trans_similarityMatrix)

        for index, item in similarityMatrixDF.iterrows():
            # print('Unsorted item')
            # print(item)

            # Sort descending, e.g. Highest similarity first (index position is retained)
            sortedItem = item.sort_values(ascending=False, kind='mergesort')
            # print('Sorted item')
            # print(sortedItem)
            # print('Top 5')

            # Get the top five items
            top5item = sortedItem[:5]
            # print(top5item)
            # print(type(top5item))

            # Each element in array represent class Edu, Skills, Personal Details, Work Experience and None.
            totalScore = np.array([0, 0, 0, 0, 0])
            for index, item in top5item.iteritems():
                # Get index info so can check what it represents.
                trainrecord = (self._trainsetDF.loc[[index]])
                # print('Index:'+str(index)+ '  item:'+ str(item))
                if (int(trainrecord['label']) == 0):
                    currentScore = [1, 0, 0, 0, 0]
                elif (int(trainrecord['label']) == 1):
                    currentScore = [0, 1, 0, 0, 0]
                elif (int(trainrecord['label']) == 2):
                    currentScore = [0, 0, 1, 0, 0]
                elif (int(trainrecord['label']) == 3):
                    currentScore = [0, 0, 0, 1, 0]
                else:
                    currentScore = [0, 0, 0, 0, 1]
                tempScore = np.vstack((totalScore, currentScore))
                totalScore = tempScore.sum(axis=0)

            # Winner label is the one with most votes. (Typically has been 100% based on sample checks)
            winnerLabel = np.argmax(totalScore)
            results.append(winnerLabel)
            # SECTION END
        return results


    def evaluate(self, approach_vsm_filename, eval=None):
        """

        :param approach_vsm_filename:
        :param eval:
        :return:
        """
        finalResults = []
        if (self._Xtrain!=None and self._Xtest!=None and self._trained==True):
            self._util.logDebug('VectorSimPredictionModel', 'Generating similarity between all training and test data')


            """
            #TODO: Refactor Section Start to End into predict method
            #SECTION START
            #Generate a similarity matrix between the vectors in training and the unseen test vectors
            sim=Similarity()
            similarityMatrix=sim.cosine_sim(self._Xtrain,self._Xtest)
            self._util.logDebug('VectorSimPredictionModel', 'Similarity matrix generated in ' + self._util.checkpointTimeTrack())
            # print('Check')
            # print(self._trainsetDF.iloc[2])
            # print(self._Xtrain[2])
            # print('Check')
            # print(self._trainsetDF.iloc[200])
            # print(self._Xtrain[200])
            # print('Check')
            # print(self._testsetDF.iloc[200])
            # print(self._Xtest[200])
            # print('Check')
            # print(self._testsetDF.iloc[2])
            # print(self._Xtest[2])



            # print('Sim matrix')
            # print(similarityMatrix)

            #Transpose it into column for easier computation
            trans_similarityMatrix=similarityMatrix.transpose()

            # print('Transpose Sim matrix')
            # print(trans_similarityMatrix)

            #Row is the test data
            #Col is the train data
            #Cell is the similarity

            similarityMatrixDF=pd.DataFrame(trans_similarityMatrix)

            for index, item in similarityMatrixDF.iterrows():
                # print('Unsorted item')
                # print(item)

                #Sort descending, e.g. Highest similarity first (index position is retained)
                sortedItem=item.sort_values(ascending=False, kind='mergesort')
                # print('Sorted item')
                # print(sortedItem)
                # print('Top 5')

                #Get the top five items
                top5item=sortedItem[:5]
                # print(top5item)
                # print(type(top5item))

                #Each element in array represent class Edu, Skills, Personal Details, Work Experience and None.
                totalScore = np.array([0, 0, 0, 0, 0])
                for index,item in top5item.iteritems():
                    #Get index info so can check what it represents.
                    trainrecord=(self._trainsetDF.loc[[index]])
                    # print('Index:'+str(index)+ '  item:'+ str(item))
                    if (int(trainrecord['label']) == 0):
                        currentScore = [1, 0, 0, 0, 0]
                    elif (int(trainrecord['label']) == 1):
                        currentScore = [0, 1, 0, 0, 0]
                    elif (int(trainrecord['label']) == 2):
                        currentScore = [0, 0, 1, 0, 0]
                    elif (int(trainrecord['label']) == 3):
                        currentScore = [0, 0, 0, 1, 0]
                    else:
                        currentScore = [0, 0, 0, 0, 1]
                    tempScore = np.vstack((totalScore, currentScore))
                    totalScore = tempScore.sum(axis=0)

                #Winner label is the one with most votes. (Typically has been 100% based on sample checks)
                winnerLabel = np.argmax(totalScore)
                self._YPredTest.append(winnerLabel)
            # SECTION END
            """
            self._YPredTest=self.predict(self._Xtest)
            self.predict_proba(self._Xtest)
            self.savePredictions(filename=approach_vsm_filename)

            pass
            # eval=Evaluator.Evaluator()
            resultsAcc = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_ACCURACY, filename=approach_vsm_filename, testDF=self.testsetDF)
            self._util.logInfo('VectorSimPredictionModel','Accuracy is ' + str(resultsAcc))

            resultsPrec = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_PRECISION, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','Precision is ' + str(resultsPrec))

            resultsRecall = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_RECALL, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','Recall is ' + str(resultsRecall))

            resultsF1 = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_F1, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','F1 is ' + str(resultsF1))

            resultsF1perclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_F1_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','F1 is ' + str(resultsF1perclass))

            resultsPrecperclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_PRECISION_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','Prec is ' + str(resultsPrecperclass))

            resultsRecallperclass = eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_RECALL_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','F1 is ' + str(resultsRecallperclass))


            resultsClass=eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_CLASSREPORT, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','Classification is \n' + resultsClass)

            resultsConfu=eval.score(y=self._Ytest, ypred=self._YPredTest, type=eval.SCORE_CONFUSIONMATRIX, filename=approach_vsm_filename)
            self._util.logInfo('VectorSimPredictionModel','Confusion is \n' + resultsConfu)

            finalResults = resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass
        else:
            self._util.logError('VectorSimPredictionModel', 'X(Training) needs to be loaded before similarity comparison!')

        return finalResults

# if __name__ == "__main__":
#     if(len(sys.argv)==4):
#         logFile = sys.argv[1]
#         testSampleFilename=sys.argv[2]
#         trainingFilename=sys.argv[3]
#
#         print('logFile:',logFile)
#         print('testSampleFilename:', testSampleFilename)
#         print('trainingFilename:', trainingFilename)
#
#         simModel=VectorSimPredictionModel(logFile=logFile)
#         simModel.loadXYtest(testSampleFilename=testSampleFilename)
#         simModel.train(trainingFilename=trainingFilename)
#         simModel.evaluate()
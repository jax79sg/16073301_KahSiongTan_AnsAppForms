"""
Inverse Answering by Classification

Input: Input labelled features appd2v...etc (Can be from BOW, TFIDF, W2V, D2V)
Input: labelled CV test examples (Must be generated from same VSM, E.g. BOW, TFIDF, W2V, D2V)
Output: predicted CV test examples
Pipe: Pipe predicted CV test examples and labelled CV test examples to Evaluator class.

For each labelled CV test example
    - Extract filename, content
    - Predict topic from content
    - Save filename, content and predicted topic (May need to rearrange the topic number)

Pass saved predicted results and labelled test examples into evaluator class for evaluation.
"""

from commons import Utilities
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing as pp
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
# from models.mlClassifierModel.KerasWrapper import KerasWrapper

def dataInsert(index, item):
    label = int(item['label'])
    vector = list(eval(item['vector']))
    # sampleDim = sampleDim + 1
    # xDim = len(vector)
    return (label, vector)

class MLClassifierPredictionModel():

    TYPE_SVM='svm'
    TYPE_MLP = 'mlp'
    TYPE_NB = 'nb'
    TYPE_NB_GAUSSIAN = 'nb_gaussian'
    TYPE_TREE='tree'
    TYPE_DUMMY='dummy'
    TYPE_XGBOOST = 'xgb'
    TYPE_SEQ='seq'
    TYPE_LOG = 'log'
    TYPE_SVC = 'svc'


    _type=None
    _model=None
    _util=None
    _Xtrain=[]
    _Ytrain=[]
    _Xtest=[]
    _Ytest=[]
    _YPredtest=[]
    _loaded=False
    _scaler=None
    testsetDF = None


    def __init__(self, logFile=None, utilObj=None, classifierType=None, classifierParams=None):

        if (utilObj != None):
            self._util = utilObj
        elif (logFile != None):
            self._util = Utilities.Utility()
            self._util.setupLogFileLoc(logFile)

        self._util.startTimeTrack()

        if (classifierType==self.TYPE_SVM):
            self._model = CalibratedClassifierCV(base_estimator=LinearSVC())
            # self._model = LinearSVC()
            self._type=classifierType
        elif (classifierType==self.TYPE_LOG):
            self._model = LogisticRegression(solver='newton-cg',n_jobs=30)
            self._type = classifierType
        elif (classifierType==self.TYPE_SVC):
            self._model = SVC(C=5,kernel='linear')
            self._type = classifierType

        elif (classifierType==self.TYPE_MLP):
            self._model = MLPClassifier()
            self._type = classifierType
        elif (classifierType==self.TYPE_NB):
            self._model = MultinomialNB()
            self._type = classifierType
        elif (classifierType==self.TYPE_NB_GAUSSIAN):
            self._model = GaussianNB()
            self._type = classifierType
        elif (classifierType == self.TYPE_TREE):
            self._model = DecisionTreeClassifier()
            self._type = classifierType
        elif (classifierType == self.TYPE_XGBOOST):
            self._model = XGBClassifier(max_depth=3, learning_rate=0.001, n_estimators=100, silent=False, objective='binary:logistic', booster='gbtree', n_jobs=32, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
            self._type = classifierType
        elif (classifierType == self.TYPE_SEQ):


            self._model = KerasWrapper(utilObj=self._util)
            self._type = classifierType

        elif (classifierType == self.TYPE_DUMMY):
            try:
                self._model = DummyClassifier()
                self._type = classifierType
            except Exception as error:
                print(error)
        else:
            self._util.logError('MLClassifierPredictionModel','An instance of the classifier needs to be passed to this class...exiting')
            exit(-1)


    def flushTestData(self):
        self._Xtest = []
        self._Ytest = []
        self._YPredtest = []
        self.testsetDF = None

    def loadXYtest(self, testSampleFilename=None):
        self._util.logDebug('MLClassifierPredictionModel','Loading test set')
        self._util.logDebug('MLClassifierPredictionModel', 'Reading CSV')
        testsetDF=pd.read_csv(testSampleFilename)
        self._util.logDebug('MLClassifierPredictionModel', 'Read CSV in ' + self._util.checkpointTimeTrack() )
        testsetDF=testsetDF.drop_duplicates(['content'])
        self.testsetDF = testsetDF
        # print('Dedup shape:',testsetDF.shape)
        self._util.logDebug('MLClassifierPredictionModel', 'Processing into list...' )
        for index,item in testsetDF.iterrows():
            label=int(item['label'])
            vector = list(eval(item['vector']))
            self._Xtest.append(vector)
            self._Ytest.append(label)
        self._util.logDebug('MLClassifierPredictionModel', 'Test set loaded in '+ self._util.checkpointTimeTrack())

    def getXtest(self):
        return self._Xtest

    def getYtest(self):
        return self._Ytest


    def train(self, trainingSampleFilename=None, findSaved=True):
        if (self._util.ifFileExists(trainingSampleFilename+'.'+self._type+'.fitted') and findSaved==True):
            self._util.logInfo('MLClassifierPredictionModel','There is a fitted model of ' + trainingSampleFilename + '... loading it now')
            self.loadFittedModel(trainingSampleFilename+'.'+self._type+'.fitted')
            self.loadFittedScaler(trainingSampleFilename+'.'+self._type+'.scaler')
        else:
            self._util.logDebug('MLClassifierPredictionModel', 'Loading training set')
            try:
                self._util.logDebug('MLClassifierPredictionModel', 'Reading CSV')
                trainsetDF = pd.read_csv(trainingSampleFilename, engine='python')
                self._util.logDebug('MLClassifierPredictionModel', 'Read CSV in ' + self._util.checkpointTimeTrack())
            except Exception as error:
                self._util.logError('MLClassifierPredictionModel','Error loading the training set,likely too big. Switching to Python engine which is stable but slower')
                trainsetDF = pd.read_csv(trainingSampleFilename, engine='python')
            xDim=0
            sampleDim=0

            multicore=True
            if (multicore==True):
                self._util.logDebug('MLClassifierPredictionModel', 'Processing into list (Multicore)...')
                from joblib import Parallel,delayed
                myset = Parallel(n_jobs=5)(delayed(dataInsert)(index, item) for index, item in trainsetDF.iterrows())
                myLabel, myVector = zip(*myset)
                self._Xtrain=myVector
                self._Ytrain=myLabel
            else:
                self._util.logDebug('MLClassifierPredictionModel', 'Processing into list (Single core)...')
                for index, item in trainsetDF.iterrows():
                    label = int(item['label'])
                    vector = list(eval(item['vector']))
                    self._Xtrain.append(vector)
                    self._Ytrain.append(label)
                    # sampleDim=sampleDim+1
                    # xDim=len(vector)
            self._util.logDebug('MLClassifierPredictionModel',
                                str(len(self._Xtrain)) + ' training samples loaded in  ' + self._util.checkpointTimeTrack())

            self._util.logInfo('MLClassifierPredictionModel','Training in progress')
            scaledXtrain=self._Xtrain
            if (self._type!=self.TYPE_NB):
                self._scaler=pp.StandardScaler().fit(self._Xtrain)
                scaledXtrain = self._scaler.transform(self._Xtrain)


            scaledXtrain, self._Ytrain=self._util.unifiedShuffle(scaledXtrain,self._Ytrain)
            self._model.fit(scaledXtrain,self._Ytrain)
            self._util.logInfo('MLClassifierPredictionModel', 'Training completed in ' + self._util.checkpointTimeTrack())
            self.saveFittedModel(trainingSampleFilename+'.'+self._type+'.fitted')
            self.saveFittedScaler(self._scaler,trainingSampleFilename+'.'+self._type+'.scaler')
            self._loaded=True

    def saveFittedScaler(self, obj, filename):
        self._util.logDebug('MLClassifierPredictionModel',
                            'Saving fitted scaler of ' + filename)
        self._util.dumpObjToFile(self._scaler, filename)
        self._util.logDebug('MLClassifierPredictionModel',
                           'Saving fitted scaler of ' + filename + ' in ' + self._util.checkpointTimeTrack())

    def saveFittedModel(self, modelFilename=None):
        self._util.logDebug('MLClassifierPredictionModel',
                            'Saving fitted model of ' + modelFilename)
        if(self._type==self.TYPE_SEQ):
            self._model.save(modelFilename)
        else:
            self._util.dumpObjToFile(self._model,modelFilename)
        self._util.logDebug('MLClassifierPredictionModel',
                           'Saving fitted model of ' + modelFilename + ' in ' + self._util.checkpointTimeTrack())


    def loadFittedModel(self,modelFilename=None):
        self._util.logDebug('MLClassifierPredictionModel',
                           'Loading fitted model of ' + modelFilename )
        if(self._type==self.TYPE_SEQ):
            self._model.load(modelFilename)

        else:
            self._model=self._util.loadFileToObj(modelFilename)
        self._util.logDebug('MLClassifierPredictionModel',
                           'Loading fitted model of ' + modelFilename + ' in ' + self._util.checkpointTimeTrack())
        self._loaded=True

    def loadFittedScaler(self,scalerFilename=None):
        self._util.logDebug('MLClassifierPredictionModel',
                           'Loading fitted scaler of ' + scalerFilename )
        self._scaler=self._util.loadFileToObj(scalerFilename)
        self._util.logDebug('MLClassifierPredictionModel',
                           'Loading fitted scaler of ' + scalerFilename + ' in ' + self._util.checkpointTimeTrack())


    def predict_proba(self,x):
        """
            Predict a topic id based on x content. The topic id is mapped to a user topic id before returning.
            :param x:
            :return:A set of probabilities refering to a topic as defined by user.
            """
        result = None
        if self._loaded:
            # print(x)
            # reshapedX=np.array(x)
            # print(reshapedX)
            # print(reshapedX.shape)
            result = self._model.predict_proba(x)
        else:
            self._util.logError('MLClassifierPredictionModel', 'Model needs to be loaded before prediction')
        return result

    def predict(self, x):
        """
        Predict a topic id based on x content. The topic id is mapped to a user topic id before returning.
        :param x:
        :return:An integer refering to a topic as defined by user.
        """
        result=None
        if self._loaded:
            # print(x)
            # reshapedX=np.array(x)
            # print(reshapedX)
            # print(reshapedX.shape)
            result=self._model.predict(x)
        else:
            self._util.logError('MLClassifierPredictionModel','Model needs to be loaded before prediction')
        return result

    def savePredictions(self, filename=None):
        """
        Save the label and prediction into a file of format
        label,prediction
        :return:
        """
        stringToSave='label,prediction\n'
        for index in range(0,len(self._YPredtest)):
            label=self._Ytest[index]
            predict=self._YPredtest[index]
            stringToSave=stringToSave+str(label)+','+str(predict)+'\n'
        self._util.saveStringToFile(stringToSave,filename=filename+'.predictions')



    def evaluate(self, approach_vsm_filename,eval=None):
        finalResults = []
        """
        :param approach_vsm_filename:
        :param eval:
        :return:resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass
        """
        if (len(self._Xtest)==len(self._Ytest) and len(self._Xtest)>0 or len(self._Ytest)>0 and self._loaded==True):
            self._util.logDebug('MLClassifierPredictionModel', 'Predicting Y for all X')
            # for x in self._Xtest:
            scaledXtest=self._Xtest
            if(self._type!=self.TYPE_NB):
                scaledXtest=self._scaler.transform(self._Xtest)

            self._YPredtest=self.predict(scaledXtest)
            # self.predict_proba(scaledXtest)
                # self._YPredtest.append(int(yPred))
            self.savePredictions(filename=approach_vsm_filename)
            print(self._YPredtest)
            self._util.logDebug('MLClassifierPredictionModel', 'Predicting Y for all X completed')
            # eval=Evaluator.Evaluator(utilObj=self._util)
            resultsAcc = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_ACCURACY, filename=approach_vsm_filename, testDF=self.testsetDF)
            self._util.logInfo('MLClassifierPredictionModel','Accuracy is ' + str(resultsAcc))

            resultsPrec = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_PRECISION, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','Precision is ' + str(resultsPrec))

            resultsRecall = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_RECALL, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','Recall is ' + str(resultsRecall))

            resultsF1 = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_F1, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','F1 is ' + str(resultsF1))

            resultsF1perclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_F1_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','F1 is ' + str(resultsF1perclass))

            resultsPrecperclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_PRECISION_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','Prec is ' + str(resultsPrecperclass))

            resultsRecallperclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_RECALL_PERCLASS, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','F1 is ' + str(resultsRecallperclass))

            resultsClass=eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_CLASSREPORT, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','Classification is \n' + resultsClass)

            resultsConfu=eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_CONFUSIONMATRIX, filename=approach_vsm_filename)
            self._util.logInfo('MLClassifierPredictionModel','Confusion is \n' + resultsConfu)

            finalResults=resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass

        else:
            self._util.logError('MLClassifierPredictionModel', 'X and Y needs to be loaded and trained before prediction!')

        return finalResults
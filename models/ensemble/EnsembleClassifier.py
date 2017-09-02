
from commons import Utilities
from models.mlClassifierModel import MLClassifierPredictionModel
from models.topicClusterModel import TopicClusteringPredictionModel
from models.vectorSimModel import VectorSimPredictionModel
import numpy as np
from models import Evaluator

class EnsembleClassifier():
    """
    Based on weighted average probabilities
    classifier	    class 1	    class 2	    class 3
classifier 1	    w1 * 0.2	w1 * 0.5	w1 * 0.3
classifier 2	    w2 * 0.6	w2 * 0.3	w2 * 0.1
classifier 3	    w3 * 0.3	w3 * 0.4	w3 * 0.3
weighted average	0.37	    0.4     	0.23

Create instance of every classifier
Fit every classifier according to their methods
Load test data
predict_proba every classifier according to their methods
Perform weighted average prob
    """

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
    TYPE_TOPIC='topic'
    TYPE_SIM='sim'

    keywordArgs={'nullkey':'nullvalue'}
    _testSize=None
    _modelList=[]
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

    def __init__(self, logFile=None, utilObj=None, **kwargs):
        if (utilObj != None):
            self._util = utilObj
        elif (logFile != None):
            self._util = Utilities.Utility()
            self._util.setupLogFileLoc(logFile)
        self._util.startTimeTrack()

        self.keywordArgs=kwargs
        print(self.keywordArgs)

    def extractKeywordArgs(self,string):
        print('Processing kwargs:',string)
        keyValueDict={'null','null'}
        keyvalues=string.split(';')
        for keyvalue in keyvalues:
            key=keyvalue.split('=')[0]
            value=keyvalue.split('=')[1]
            keyValueDict[key]=value
        return keyValueDict


    def loadClassifiers(self,classifierTypeListStr=None):

        classifierTypeList=classifierTypeListStr.split(',')
        for classifierType in classifierTypeList:
            predictionModel=None
            if (classifierType==self.TYPE_SIM):
                predictionModel = VectorSimPredictionModel.VectorSimPredictionModel(utilObj=self._util)
            elif(classifierType==self.TYPE_TOPIC):
                predictionModel = TopicClusteringPredictionModel.TopicClusteringPredictionModel(utilObj=self._util,ldaModelFilename=self.keywordArgs['ldaModelFilename'])
            else:
                predictionModel = MLClassifierPredictionModel.MLClassifierPredictionModel(utilObj=self._util,classifierType=classifierType)

            self._modelList.append(predictionModel)


    def trainClassifiers(self,appd2vFilename=None):
        for classifier in self._modelList:
            classifier.train(appd2vFilename)

    def loadXYtest(self,sampleLabelledTestFilename=None):
        for classifer in self._modelList:
            classifer.flushTestData()
            classifer.loadXYtest(sampleLabelledTestFilename)
            self._testSize=len(classifer.getXtest())
            self._Ytest=classifer.getYtest()

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

    def evaluate(self, approach_vsm_filename=None, eval=None, weight=[1,1,1,1,1], classweight=None):
        weightStr=''.join(map(str, weight))
        summedWeight=np.sum(weight)
        summedClassweight=np.sum(classweight)
        predictionProbaList=[]
        classifierIndex=0
        for classifer in self._modelList:
            predictionProba=classifer.predict_proba(classifer.getXtest())
            if(classweight!=None):
                #Adjust class weight, give certain classes higher weight since this classifier seems to perform better at that.
                def funcy(a):
                    sum = np.sum(a)
                    result = a / sum
                    return result

                classifierWeight=classweight[classifierIndex]
                prediction = predictionProba
                prediction = np.array(prediction)
                # print('Prediction', prediction)
                # weight = [2, 1, 1, 1]
                prediction = prediction * classifierWeight
                # print('Multiply:', prediction)
                prediction = np.apply_along_axis(funcy, 1, prediction)
                # print('Final Pred:', prediction)
                predictionProba=prediction

            predictionProbaList.append(predictionProba)
            classifierIndex=classifierIndex+1
        newPredictions=[]

        noOfClassifiers=len(predictionProbaList)
        for xIndex in range(0,self._testSize):
            #For each sample, perform weighted probability.
            currentIndex=0
            totalProba = 0
            for predictionProba in predictionProbaList:
                #Multiply all classes by weight for classifier
                currentProba=predictionProba[xIndex]*weight[currentIndex]
                totalProba=totalProba+currentProba
                currentIndex=currentIndex+1
            newProba=totalProba/summedWeight
            result=np.argmax(newProba)
            newPredictions.append(result)

        self._YPredtest=newPredictions
        self.savePredictions(filename=approach_vsm_filename+'_'+weightStr)
        self._util.logDebug('EnsembleClassifier', 'Predicting Y for all X completed')
        # eval=Evaluator.Evaluator(utilObj=self._util)
        resultsAcc = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_ACCURACY,
                                filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Accuracy is ' + str(resultsAcc))

        resultsPrec = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_PRECISION,
                                 filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Precision is ' + str(resultsPrec))

        resultsRecall = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_RECALL,
                                   filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Recall is ' + str(resultsRecall))

        resultsF1 = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_F1, filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'F1 is ' + str(resultsF1))

        resultsF1perclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_F1_PERCLASS,
                                       filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'F1 is ' + str(resultsF1perclass))

        resultsPrecperclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_PRECISION_PERCLASS,
                                         filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Prec is ' + str(resultsPrecperclass))

        resultsRecallperclass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_RECALL_PERCLASS,
                                           filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Recall is ' + str(resultsRecallperclass))

        resultsClass = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_CLASSREPORT,
                                  filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Classification is \n' + resultsClass)

        resultsConfu = eval.score(y=self._Ytest, ypred=self._YPredtest, type=eval.SCORE_CONFUSIONMATRIX,
                                  filename=approach_vsm_filename+'_'+weightStr)
        self._util.logInfo('EnsembleClassifier', 'Confusion is \n' + resultsConfu)

        finalResults = resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass



util=Utilities.Utility()
util.setupLogFileLoc(logFile='/u01/bigdata/02d_d2vModel1/features/log_ensemble.log')
util.setupTokenizationRules('removeStopwords,toLowercase,replaceSlash')
ensemble=EnsembleClassifier(utilObj=util,ldaModelFilename='/u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model')
# ensemble.loadClassifiers(classifierTypeListStr='topic,sim,log,svm,mlp')
# ensemble.loadClassifiers(classifierTypeListStr='sim,log,svm,mlp')
# ensemble.loadClassifiers(classifierTypeListStr='log,svm,mlp')
ensemble.loadClassifiers(classifierTypeListStr='log,svm')
ensemble.trainClassifiers(appd2vFilename='/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v100min1.features')
ensemble.loadXYtest(sampleLabelledTestFilename='/u01/bigdata/02d_d2vModel1/features/cvTestW2v100min1.features')
eval = Evaluator.Evaluator(utilObj=util)
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,3,2,2,1],classweight=[[2,1,3,2],[1,2,3,1],[1,1,1,1],[1,1,2,1.5],[1,2,1,2]])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,3,2,2,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[1,1,1,1,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[5,3,2,2,1],classweight=[[5,1,3,2],[3,2,3,1],[1,1,1,1],[1,1,2,1.5],[1,2,1,2]])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,2,2,1],classweight=[[3,2,3,1],[1,1,1,1],[1,1,2,1.5],[1,2,1,2]])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,3,2],classweight=[[1,1,1,1],[1,1,2,1.5],[1,2,1,2]])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,1,1],classweight=[[1,1,1,1],[1,1,2,1.5],[3,4,3,1]])
ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[1,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,1,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[3,2,2])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[1,3,1,1,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[2,2,3,3,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[1,1,3,1,1])
# ensemble.evaluate('/u01/bigdata/02d_d2vModel1/features/ENSEMBLE_W2V100min1', eval,[2,2,3,2,2])




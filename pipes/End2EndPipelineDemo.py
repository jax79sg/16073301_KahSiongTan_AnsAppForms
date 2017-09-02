
"""
    1. Lookup a CV (raw)
    2. Convert to txt
    3. Split into sections
    . Convert sections to vector format
"""
import glob
from subprocess import run
import subprocess
from data.extractFromCV import CVReader
from features.W2V import W2V
from features.D2V import D2V
from features.BOW import BOW
from features.TFIDF import TFIDF
from commons import Utilities
from models.vectorSimModel.VectorSimPredictionModel import VectorSimPredictionModel
from models.topicClusterModel.TopicClusteringPredictionModel import TopicClusteringPredictionModel
from models.mlClassifierModel.MLClassifierPredictionModel import MLClassifierPredictionModel
import pandas as pd
import numpy as np

#For topic
#python3 End2EndPipelineDemo.py '/u01/bigdata/02d_d2vModel1/testCV/logs/testCV_topic.log' '/u01/bigdata/02d_d2vModel1/testCV' 'ldaModelFilename=/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/CvLda4TopicModel.model' vectorSpaceModelType=topic
#python3 End2EndPipelineDemo.py '/u01/bigdata/02d_d2vModel1/testCV/logs/testCV_sim.log' '/u01/bigdata/02d_d2vModel1/testCV'  'predictionTrainingFilename=/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/features/appD2vTrainW2v100min1.features'  vectorSpaceModelType=sim vectorSpaceModelFilename=cvW2v100min1VectorSpaceModel.model saveResultsFilename=/u01/bigdata/02d_d2vModel1/testCV/results/resultsSim.csv
#For sim

class End2EndPipelineDemo():

    util=None
    cvDataframe =None
    vectorSpaceModel=None
    vectorSpaceModelType=None
    predictionModelType = None
    predictionModel = None
    predictionTrainingFilename=None
    vectorSpaceModelFilename=None
    saveResultsFilename=None

    locationOfCVs=None
    def __init__(self, locationOfCVs=None,utilObj=None,logFilename=None, **kwargs):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)

        self.cvDataframe= pd.DataFrame(columns=('filename', 'content', 'vector', 'proba', 'label'))  # Structure of the csv to be saved
        self.predictionModelType = kwargs['predictionModelType']

        if (self.predictionModelType=='topic'):
            self.ldaModelFilename=kwargs['ldaModelFilename']
        elif(self.predictionModelType=='sim'):
            self.vectorSpaceModelType=kwargs['vectorSpaceModelType']
            self.vectorSpaceModelFilename = kwargs['vectorSpaceModelFilename']
            self.predictionTrainingFilename = kwargs['predictionTrainingFilename']
        else:
            self.vectorSpaceModelType=kwargs['vectorSpaceModelType']
            self.vectorSpaceModelFilename = kwargs['vectorSpaceModelFilename']
            self.predictionTrainingFilename = kwargs['predictionTrainingFilename']

        self.saveResultsFilename=kwargs['saveResultsFilename']


        self.locationOfCVs=locationOfCVs


        pass

    def startDemo(self):
        self.processRaws()
        self.splitCVs()

        self.loadVectorModel()
        self.loadPredModel()
        if(self.predictionModelType!='topic'):
            self.inferVectors()
        self.predictQuestions()
        self.saveResults()


    def saveResults(self):
        self.util.logInfo('End2EndPipelineDemo',self.cvDataframe.to_string())
        self.cvDataframe.to_csv(self.saveResultsFilename, ',', mode='w',header=True, index=False, columns=('filename','label','proba'))


    def loadPredModel(self):
        if (self.predictionModelType.lower()=='sim'):
            self.predictionModel=VectorSimPredictionModel(utilObj=self.util)
            self.predictionModel.train(trainingFilename=self.predictionTrainingFilename)
            pass
        elif(self.predictionModelType.lower()=='topic'):
            self.predictionModel = TopicClusteringPredictionModel(utilObj=self.util,ldaModelFilename=self.ldaModelFilename)
            pass
        elif (self.predictionModelType.lower() == 'log'):
            self.predictionModel = MLClassifierPredictionModel(utilObj=self.util,classifierType='log')
            self.predictionModel.train(self.predictionTrainingFilename)
            pass
        elif (self.predictionModelType.lower() == 'svm'):
            self.predictionModel = MLClassifierPredictionModel(utilObj=self.util,classifierType='svm')
            self.predictionModel.train(self.predictionTrainingFilename)
            pass
        elif (self.predictionModelType.lower() == 'mlp'):
            self.predictionModel = MLClassifierPredictionModel(utilObj=self.util,classifierType='mlp')
            self.predictionModel.train(self.predictionTrainingFilename)
            pass

    def predictQuestions(self):
        for index, item in self.cvDataframe.iterrows():
            pass
            X=None
            if (self.predictionModelType=='topic'):
                X=[]
                X.append(item['content'])
            else:
                X = []
                X.append(item['vector'])
                X=np.array(X)
            # predictionProb=self.predictionModel.predict_proba(X)
            prediction=self.predictionModel.predict(X)
            self.cvDataframe.set_value(index, 'label', prediction)
            # self.cvDataframe.set_value(index, 'proba', ' '.join(str(e) for e in predictionProb))

    def inferVectors(self):
        for index,item in self.cvDataframe.iterrows():
            pass
            content=item['content']
            vector=self.vectorSpaceModel.inferVector(content)
            self.cvDataframe.set_value(index,'vector',vector)
        pass

    def loadVectorModel(self):
        if(self.predictionModelType!='topic'):
            if (self.vectorSpaceModelType=='w2v'):
                self.vectorSpaceModel=W2V(utilObj=self.util)
                pass
            elif(self.vectorSpaceModelType=='d2v'):
                self.vectorSpaceModel = D2V(utilObj=self.util)
                pass
            elif(self.vectorSpaceModelType=='tf'):
                self.vectorSpaceModel = BOW(utilObj=self.util)
                pass
            elif(self.vectorSpaceModelType=='tfidf'):
                self.vectorSpaceModel = TFIDF(utilObj=self.util)
                pass
            else:
                self.util.logError('End2EndPipelineDemo','No vector model loaded! Cannot continue!')
                exit(-1)
            self.vectorSpaceModel.loadVectorSpaceModel(self.vectorSpaceModelFilename)

    def splitCVs(self):
        for convertedTxt in (glob.iglob(self.locationOfCVs + '/*.txt')):
            cvReader=CVReader.CVReader(convertedTxt)
            sectionList=cvReader.getParagraphs()
            #Save results to dataframe
            sectionCounter=0
            for section in sectionList:
                newrow=pd.DataFrame(data={'filename':[convertedTxt+'_'+str(sectionCounter).zfill(2)],'content':[section]})
                self.cvDataframe = self.cvDataframe.append(newrow)
                sectionCounter=sectionCounter+1
                self.util.saveStringToFile(section,convertedTxt+'_'+str(sectionCounter).zfill(2))

        pass

        self.cvDataframe=self.cvDataframe.reset_index()

    def processRaws(self):
        for originalRaw in (glob.iglob(self.locationOfCVs + '/*.*')):
            self._convertToText(originalRaw)
            pass

    def _convertToText(self,filename):
        fileExt=filename.split('.')[-1]
        if(fileExt.lower()=='docx'):
            pass
            run (["docx2txt",filename],stdout=subprocess.PIPE)
        elif(fileExt.lower()=='pdf'):
            run(["pdftotext", "-layout",filename], stdout=subprocess.PIPE)
            pass

        pass

util=Utilities.Utility()
util.setupLogFileLoc('/u01/bigdata/02d_d2vModel1/testCV/logs/testCV_topic.log')

# pipeline=End2EndPipelineDemo(locationOfCVs='/u01/bigdata/02d_d2vModel1/testCV',utilObj=util,saveResultsFilename='/u01/bigdata/02d_d2vModel1/testCV/results/topicResults.csv',ldaModelFilename='/u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model', predictionModelType='topic')
# pipeline.startDemo()
pipeline=End2EndPipelineDemo(locationOfCVs='/u01/bigdata/02d_d2vModel1/testCV',utilObj=util,saveResultsFilename='/u01/bigdata/02d_d2vModel1/testCV/results/simResults.csv', predictionModelType='sim',vectorSpaceModelType='w2v',vectorSpaceModelFilename='/u01/bigdata/02d_d2vModel1/cvW2v100min1VectorSpaceModel.model', predictionTrainingFilename='/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v100min1.features')
pipeline.startDemo()
pipeline=End2EndPipelineDemo(locationOfCVs='/u01/bigdata/02d_d2vModel1/testCV',utilObj=util,saveResultsFilename='/u01/bigdata/02d_d2vModel1/testCV/results/simTFIDFResults.csv', predictionModelType='sim',vectorSpaceModelType='tfidf',vectorSpaceModelFilename='/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_5000.model', predictionTrainingFilename='/u01/bigdata/02d_d2vModel1/features/appD2vTrainTfidf5000.features')
pipeline.startDemo()
pipeline=End2EndPipelineDemo(locationOfCVs='/u01/bigdata/02d_d2vModel1/testCV',utilObj=util,saveResultsFilename='/u01/bigdata/02d_d2vModel1/testCV/results/simD2VResults.csv', predictionModelType='sim',vectorSpaceModelType='d2v',vectorSpaceModelFilename='/u01/bigdata/02d_d2vModel1/cvD2v100VectorSpaceModel.model', predictionTrainingFilename='/u01/bigdata/02d_d2vModel1/features/appD2vTrainD2v100.features')
pipeline.startDemo()
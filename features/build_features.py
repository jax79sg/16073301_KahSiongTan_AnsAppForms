"""
Used to build vectors from the datasets

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
from features import BOW
from features import W2V
from features import D2V
from features import TFIDF

import pandas as pd
import glob
import sys
import numpy

class FeatureBuilder():

    util=None
    vsmModel=None
    VSM_D2V='d2v'
    VSM_W2V='w2v'
    VSM_BOW='bow'
    VSM_TFIDF='tfidf'


    def __init__(self, logFilename=None, utilObj=None):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)
            self.util.startTimeTrack()
        pass

    def _setVSMType(self, vsmType=None):
        if (vsmType == None):
            self.util.logError('buildFeatures', 'No Vector Space Model provided..cannot proceed')
            exit(-1)
            pass
        elif (vsmType == self.VSM_BOW):
            self.vsmModel = BOW.BOW(utilObj=self.util)
        elif (vsmType == self.VSM_W2V):
            self.vsmModel = W2V.W2V(utilObj=self.util)
        elif (vsmType == self.VSM_TFIDF):
            self.vsmModel = TFIDF.TFIDF(utilObj=self.util)
        elif (vsmType == self.VSM_D2V):
            self.vsmModel = D2V.D2V(utilObj=self.util)
        else:
            self.util.logError('buildFeatures', vsmType+' yet to be implemented.. cannot proceed')

    def buildMultifeatures(self, srcFolderList=None, labelList=None, vsmTypeList=None, vsmModelFilenameList=None, dstFilename=None):
        """
        This builds a multi feature set. Basically concatenating into a single vector for all sorts of features.
        :param srcFolderList:
        :param labelList:
        :param vsmTypeList:
        :param vsmModelFilenameList:
        :param dstFilename:
        :return:
        """
        numpy.set_printoptions(threshold=numpy.nan)
        if (labelList == 'None'):
            labelList = None
        else:
            labelList = labelList.split(',')
        self.vsmModel = None
        # Create an empty file with header
        featuresDF = pd.DataFrame(columns=('filename', 'content', 'label', 'vector'))
        featuresDF.to_csv(dstFilename, ',', mode='w', header=True, index=False,
                          columns=['filename', 'content', 'label', 'vector'])
        if (srcFolderList == None or vsmTypeList == None or vsmModelFilenameList == None or dstFilename == None):
            self.util.logError('buildFeatures', 'All arguments to buildFeatures need to be provided..cannot proceed')
            exit(-1)
        vsmTypeList=vsmTypeList.split(',')
        vsmModelFilenameList=vsmModelFilenameList.split(',')
        vsmModelList=[]
        for index in range(0, len(vsmTypeList)):
            self._setVSMType(vsmType=vsmTypeList[index])
            self.vsmModel.loadVectorSpaceModel(vsmModelFilenameList[index])
            vsmModelList.append(self.vsmModel)

        # Means this dataset has no labels
        folderCount = 0
        rowCount = 0
        for folder in srcFolderList.split(','):
            self.util.logDebug('buildFeatures', 'Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.*')):

                content = None
                label = None
                vector = None

                if (len(self.util.tokenize(self.util.readFileContent(filename=filename))) > 10):
                    # If there is less than 10 words, than the feature is not good. Skip.
                    content = self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename)),
                                                    ' ')
                    label = None
                    if (labelList != None):
                        # Means this dataset has labels
                        label = labelList[folderCount]

                    #For each type of representation, infer the vector and concatenate to existing vectors.
                    totalTypes=len(vsmTypeList)
                    vector=[]
                    for index in range(0,totalTypes):
                        self.vsmModel=vsmModelList[index]
                        currentVector = self.vsmModel.inferVector(content)
                        vector.extend(currentVector)
                        # print('Added '+ str(vsmTypeList[index]) + ' new size is ' + str(len(vector)))
                        # print('shape:',vector.shape)
                    newrow = pd.DataFrame(
                        data={'filename': [filename], 'content': [content], 'label': [label], 'vector': [vector]})
                    featuresDF = featuresDF.append(newrow)
                    if (rowCount % 100 == 0):
                        featuresDF.to_csv(dstFilename, ',', mode='a', header=False, index=False,
                                          columns=['filename', 'content', 'label', 'vector'])
                        featuresDF = featuresDF[0:0]
                        self.util.logDebug('buildFeatures', str(rowCount) + ' records saved in ' + dstFilename)
                    rowCount = rowCount + 1

            folderCount = folderCount + 1
            # Save remaining features
            featuresDF.to_csv(dstFilename, ',', mode='a', header=False, index=False,
                              columns=['filename', 'content', 'label', 'vector'])
            self.util.logDebug('buildFeatures', 'Final records saved in ' + self.util.checkpointTimeTrack())
        pass

    def buildFeatures(self, srcFolderList=None, labelList=None, vsmType=None, vsmModelFilename=None, dstFilename=None):
        """
        Load all the files in the srcFolderList
        For each file, read the content, parse it through the tokeniser, generate the vector
        Save the record
        Finally save the file
        :param srcFolderList:Folder containing cvd2v files
        :param labelList: List of labels that corresponds in order to the list of srcFolderList (Format 1,2,3..)
        :param vsmType: 'd2v' or 'w2v' or 'bow' or 'tfidf'
        :param vsmModelFilename: the vsm model filename corresponding to the vsmtype.
        :param dstFilename: Where to save the feature files (This output can be used directory in the prediction models)
        :return:
        """
        numpy.set_printoptions(threshold=numpy.nan)
        if (labelList=='None'):
            labelList=None
        else:
            labelList=labelList.split(',')
        self.vsmModel=None
        #Create an empty file with header
        featuresDF = pd.DataFrame(columns=('filename', 'content', 'label', 'vector'))
        featuresDF.to_csv(dstFilename, ',', mode='w', header=True, index=False,columns=['filename', 'content', 'label', 'vector'])
        if (srcFolderList==None or vsmType==None or vsmModelFilename==None or dstFilename==None):
            self.util.logError('buildFeatures','All arguments to buildFeatures need to be provided..cannot proceed')
            exit(-1)
        self._setVSMType(vsmType=vsmType)
        self.vsmModel.loadVectorSpaceModel(vsmModelFilename)

        #Means this dataset has no labels
        folderCount=0
        rowCount=0
        for folder in srcFolderList.split(','):
            self.util.logDebug('buildFeatures','Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.*')):

                content=None
                label=None
                vector=None

                if(len(self.util.tokenize(self.util.readFileContent(filename=filename)))>10):
                    #If there is less than 10 words, than the feature is not good. Skip.
                    content = self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename)), ' ')
                    label=None
                    if (labelList!=None):
                        #Means this dataset has labels
                        label=labelList[folderCount]
                    vector=self.vsmModel.inferVector(content)
                    # print('shape:',vector.shape)
                    newrow = pd.DataFrame(data={'filename': [filename], 'content': [content], 'label': [label], 'vector': [vector]})
                    featuresDF = featuresDF.append(newrow)
                    if (rowCount%100==0):
                        featuresDF.to_csv(dstFilename, ',', mode='a', header=False, index=False,columns=['filename', 'content', 'label', 'vector'])
                        featuresDF = featuresDF[0:0]
                        self.util.logDebug('buildFeatures', str(rowCount)+' records saved in ' + dstFilename )
                    rowCount=rowCount+1

            folderCount = folderCount + 1
            #Save remaining features
            featuresDF.to_csv(dstFilename, ',', mode='a', header=False, index=False,columns=['filename', 'content', 'label', 'vector'])
            self.util.logDebug('buildFeatures','Final records saved in ' + self.util.checkpointTimeTrack())
        #  = None,  = None,  = None,  = None,  = None
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainBOW10000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_10000.model' '/u01/bigdata/02d_d2vModel1/features/appd2vTrainBOW10000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainBOW5000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/appd2vTrainBOW5000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainD2v100features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'd2v' '/u01/bigdata/02d_d2vModel1/cvD2v100VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainD2v100.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainD2v300features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'd2v' '/u01/bigdata/02d_d2vModel1/cvD2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainD2v300.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainTfidf5000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'tfidf' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainTfidf5000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainTfidf10000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'tfidf' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_10000.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainTfidf10000.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainW2v100min1features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v100min1.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainW2v100min2features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v100min2.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainW2v300min1features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v300min1.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainW2v300min2features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/appD2vTrainW2v300min2.features'



#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainBOW10000features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_10000.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainBOW10000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainBOW5000features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainBOW5000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainD2v100features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'd2v' '/u01/bigdata/02d_d2vModel1/cvD2v100VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainD2v100.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainD2v300features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'd2v' '/u01/bigdata/02d_d2vModel1/cvD2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainD2v300.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainTfidf5000features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'tfidf' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainTfidf5000.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainTfidf10000features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'tfidf' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_10000.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainTfidf10000.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainW2v100min1features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v100min1.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainW2v100min2features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v100min2.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainW2v300min1features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v300min1.features'
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainW2v300min2features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v300min2.features'

#test
#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_test.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0a,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1a,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2a,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3a' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/test.features'

#python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTestClient291W2v300min1features.log' '/u01/bigdata/03b_raters/client291/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/client291/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/client291/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/client291/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100min1VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient291TrainW2v100min1.features' 'removeStopwords,toLowercase,replaceSlash'

if __name__ == "__main__":
    if(len(sys.argv)==8):
        logLoc = sys.argv[1]
        srcFolderList = sys.argv[2]
        labelList = sys.argv[3]
        vsmType = sys.argv[4]
        vsmModelFilename = sys.argv[5]
        dstFilename = sys.argv[6]
        tokenRules = sys.argv[7]

        print('logloc: ', logLoc)
        print('srcFolderList: ',srcFolderList)
        print('labelList: ', labelList)
        print('vsmType: ', vsmType)
        print('vsmModelFilename: ', vsmModelFilename)
        print('dstFilename: ', dstFilename)
        print('tokenRules: ', tokenRules)

        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logLoc)
        util.setupTokenizationRules(tokenRules)

        fbuild = FeatureBuilder(utilObj=util)
        # fbuild.buildFeatures(srcFolderList=srcFolderList, labelList=labelList, vsmType=vsmType, vsmModelFilename=vsmModelFilename, dstFilename=dstFilename)
        fbuild.buildMultifeatures(srcFolderList=srcFolderList, labelList=labelList, vsmTypeList=vsmType, vsmModelFilenameList=vsmModelFilename, dstFilename=dstFilename)

    else:
        print("No arguments provided..")
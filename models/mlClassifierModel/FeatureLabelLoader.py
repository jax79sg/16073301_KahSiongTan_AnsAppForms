"""
- Single training appd2v doc2vec model
    - Generate a doc2vec vector space model from appd2v
        - Dump all training app2v (regardless of categories) to generate model (Completed)
    - Create training data
        - training appd2v and label
            - Infer a document vector for each appd2v, store the vector and label. (Completed)
    - Create training test data
        - Use the 80 CVs that will be labelled. (In the labelled CSV after majority vote, discard the 0,0,0,0 labels)
        - For each item in the majority vote
            - Infer a document vector for each cvd2v, store the vector and the respective label.

- Single training cvd2v doc2vec model
    - Generate a doc2vec model
        - Dump all training cvd2v to generate model
    - Create training data
        - training appd2v and label
            - Infer a document vector for each appd2v, store the vector and label.
    - Create test data
        - Use the 80 CVs that will be labelled.
        - Infer a document vector for each cvd2v, store the vector

- Single bag of words model
    - Generate a BOW model
        - Dump all training cvd2v to generate model
    - Create training data
        - training appd2v and label
            - Infer a high dimension vector for each app2v, store the vector and label
    - Create test data
        - Use the 80 CVs that will be labelled.
        - Infer a high dimension vector for each app2v, store the vector and label


"""

from models.doc2vec import doc2vecFrame
import numpy as np
from commons import Utilities
import glob
import pandas as pd

class FeatureLabelLoader():

    util=None

    def __init__(self, utilObj=None, logFilename=None):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)

        pass

    def unifiedShuffle(self,npA, npB):
        """
        Shuffle numpyA and numpyB in the same order
        :param npA:
        :param npB:
        :return:
        """
        assert len(npA)==len(npB)
        p=np.random.permutation(len(npA))
        return npA[p],npB[p]

    def generateTrainLabelledD2VFeatures(self, folderTrainAppd2v=None, label=None, d2vVectorSpaceModel=None, maxSize=10, destFeatureFile=None):
        """
        For each appd2v in folderTrainAppd2v, infer a document vector from the d2vVectorSpaceModel.
        Put the results in dataframe of structure
            - filename, documentvector, label
        Save the dataframe into a csv in destFeatureFile
        :param folderTrainAppd2v: /u01/bigdata/01b_d2v/032/ZZZ/doc2vecZZZ/train,any other folders
        :param label: ZZZ, Any other labels
        :param maxSize: Use the max of all labels, in this case 15000 to ensure balance.
        :return:
        """
        d2vFeatureDF = pd.DataFrame(columns=('appd2vfilename', 'd2vvector', 'label'))

        d2vFrame=doc2vecFrame.doc2vecFrame(utilObj=self.util)
        d2vFrame.loadModel(d2vVectorSpaceModel)
        nthFolder=0
        counter=0
        maxSizePerFolder = int(maxSize /len(folderTrainAppd2v.split(',')))
        for folderTrain in folderTrainAppd2v.split(','):
            self.util.logDebug('generateTrainLabelledD2VFeatures', 'Processing from ' + folderTrain)
            for cvsectionFilename in sorted(glob.iglob(folderTrain + '/*.*')):
                contentStr=self.util.readFileContent(cvsectionFilename)
                vector=d2vFrame.inferVector(contentStr)
                currentLabel=label.split(',')[nthFolder]
                newrow = pd.DataFrame(data={'appd2vfilename': [cvsectionFilename], 'd2vvector': [vector], 'label': [currentLabel]})
                d2vFeatureDF = d2vFeatureDF.append(newrow)
                if (counter%100==0):
                    d2vFeatureDF.to_csv(destFeatureFile, ',', mode='a',header=False, index=False, columns=('appd2vfilename', 'd2vvector','label'))
                    d2vFeatureDF = d2vFeatureDF[0:0]
                    self.util.logDebug('generateTrainLabelledD2VFeatures', 'Saving!')
                counter=counter+1
                if (counter>=maxSizePerFolder):
                    break
            nthFolder=nthFolder+1
            counter=0
            d2vFeatureDF.to_csv(destFeatureFile, ',', mode='a', header=False, index=False,
                                columns=('appd2vfilename', 'd2vvector', 'label'))
            self.util.logDebug('generateTrainLabelledD2VFeatures', 'Final save for folder!')
        pass

    def loadTrainLabelledD2VFeatures(self,srcFeatureFile=None ):
        """
        Load the dataframe from srcFeatureFile
        For each row,
            Extract vector and label
            Add to numpy array npX and npY
        :param srcFeatureFile:
        :return:
        """
        npX=None
        npY=None

        return npX, npY

    def loadBagOfWordsFeatureLabels(self):
        pass

    def loadTestDoc2vecFeatures(self,d ,modelFilename=None):
        """

        :param modelFilename:
        :return:
        """
        pass

    def loadTrainDoc2vecFeatureLabels(self, modelFilename=None, label=None):
        npX=None
        npY=None

        d2vModel=doc2vecFrame.doc2vecFrame(utilObj=self.util)
        d2vModel.loadModel(modelFilenamePath=modelFilename)
        npX=d2vModel.getNumpVectors()
        # d2vDF=d2vModel.getAllDocVecs()
        # print(npX.shape)
        npY=np.full((npX.shape[0],1),label)
        # for index, row in d2vDF.iterrows():
        #     doctag=label
        #     docvec=np.array(row['docvec'])
        #     npX=np.append(docvec,axis=0)
        #     npY=np.append(doctag,axis=0)
        return npX,npY
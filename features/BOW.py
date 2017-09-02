"""
Use sklearn countvectorizer
Setup corpus (all the training cvd2v)
Setup fit transform (One time_
Setup transform (Prediction of sentences)
"""
from sklearn.feature_extraction.text import CountVectorizer
from commons import Utilities
from sklearn.externals import joblib
import glob
import numpy as np
class BOW():


    corpus=None
    model=None
    util=None
    fitted=False

    def __init__(self, logFilename=None, utilObj=None, maxDim=None, ngram=1):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)
        self.util.logDebug('BOW', 'Initialising BOW')
        self.util.startTimeTrack()
        self.model=CountVectorizer(max_features=maxDim, ngram_range=(1,ngram))
        self.util.logDebug('BOW',
                             'Initialising BOW completed ' + self.util.stopTimeTrack())
        pass


    def saveVectorSpaceModel(self, dstFilename):
        joblib.dump(self.model, dstFilename)


    def loadVectorSpaceModel(self, srcFilename):
        self.util.logDebug('BOW','Loading BOW vector space model from ' + srcFilename)
        self.model = joblib.load(srcFilename)
        self.fitted=True
        self.util.logDebug('BOW', 'Loading BOW vector space model completed in ' + self.util.checkpointTimeTrack())
        pass



    def buildCorpus(self, folderListOfCorpus=None, ngram=1, maxdocs=-1, dstFilename=None, maxDim=None):
        """
        For each folder
            for each cvd2v in in folder
                Get tokens from Utility tokenise and then form into a string
                Append string into a list (This forms a document)
        Run vectoriser fit
        :param dstFilename:
        :param folderListOfCorpus:
        :param ngram:
        :param maxdocs: Max number of documents to look at
        :return:
        """
        self.util.logDebug('BOW', 'Building and fitting corpus ')
        self.corpus=[]
        maxDocPerFolder=int(maxdocs/len(folderListOfCorpus.split(',')))
        docCounter=0
        for folder in folderListOfCorpus.split(','):
            self.util.logDebug('BOW', 'Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.cvd2v')):
                if (docCounter <= maxDocPerFolder):
                    fileContent=self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename),removeStopwords=True,toLowercase=True,replaceSlash=True,flatEmail=True,flatMonth=True,flatNumber=True,lemmatize=True), ' ')
                    self.corpus.append(fileContent)
                    docCounter=docCounter+1
                else:
                    docCounter=0
                    break

        self.util.logDebug('BOW', 'Corpus loaded in ' + self.util.stopTimeTrack())
        self.model.fit(self.corpus)
        self.fitted=True
        self.util.logDebug('BOW', 'Corpus fitted with ' + str(len(self.model.vocabulary_)) + ' vocab words in ' + self.util.stopTimeTrack())
        self.saveVectorSpaceModel(dstFilename=dstFilename)
        self.util.logDebug('BOW', 'Model saved in ' + self.util.stopTimeTrack())


    def inferVector(self, strList):
        results=None
        if(self.fitted==False):
            self.util.logError('BOW', 'The corpus has yet to be loaded, this operation cannot proceed')
            exit(-1)
        else:
            results= self.model.transform([strList])
            # print(results.toarray())
        return np.array(results.toarray()).tolist()[0]
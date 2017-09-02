"""
Topic modelling: LDA
"""
import glob
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from commons import Utilities
import pyLDAvis.gensim
from collections import defaultdict
import os
import pickle

class LDA():

    corpus=None
    model=None
    dictionary=None
    util=None
    loaded=False
    topicLabelling=defaultdict(int)

    def __init__(self, utilObj=None, logfilename=None):
        if (utilObj != None):
            self.util = utilObj
        elif (logfilename != None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logfilename)

        self.util.startTimeTrack()


    def labelTopics(self, modelFilename):


        if(os.path.exists(modelFilename+'.label')):
            f=open(modelFilename+'.label',"rb")
            self.topicLabelling=pickle.load(f)
            f.close()
        else:
            #Label file not available, performing manual labelling. (One time operation)
            topics=self.model.show_topics(num_topics=100,num_words=20)
            print('You will be shown a series of words and asked to label the topic in the form of an integer\n')
            for topic in topics:
                print('The words affliated to this topic is as follows\n',topic[1])
                print('\033[92m'+'Please label as one of these \n(0) EDUCATION\n(1) SKILLS\n(2) PERSONAL DETAILS\n(3) WORK EXPERIENCE'+'\033[0m')
                mappedTopicInt=input('Please enter a new integer for this topic: ')
                self.topicLabelling[topic[0]]=mappedTopicInt
            f=open(modelFilename+'.label',"wb")
            pickle.dump(self.topicLabelling, f)
            f.close()

    def buildCorpus(self, folderListOfCorpus=None, maxdocs=-1):
        """
        For each folder
            for each cvd2v in in folder
                Get tokens from Utility tokenise and then form into a string
                Append string into a list (This forms a document)
        """
        self.util.logDebug('LDA', 'Building and fitting corpus ')
        documentList=[]
        maxDocPerFolder=int(maxdocs/len(folderListOfCorpus.split(',')))
        docCounter=0
        for folder in folderListOfCorpus.split(','):
            self.util.logDebug('LDA', 'Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.cvd2v')):
                if (docCounter <= maxDocPerFolder):
                    fileContent=self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename),removeStopwords=True,toLowercase=True,replaceSlash=True,flatEmail=True,flatMonth=True,flatNumber=True,lemmatize=True), ' ')
                    documentList.append(fileContent)
                    docCounter=docCounter+1
                else:
                    docCounter=0
                    break

        self.util.logDebug('LDA', str(len(documentList)) + ' documents loaded in ' + self.util.stopTimeTrack())
        texts = [[word for word in document.lower().split()] for document in documentList]
        self.util.logDebug('LDA', 'No of vocab words: ' + str(len(texts)))
        self.util.logDebug('LDA', 'Text example: ' + str(texts[0]))
        self.dictionary = Dictionary(texts)


        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.util.logDebug('LDA', 'Corpus built in ' + self.util.stopTimeTrack())

    def trainModel(self, noOfTopics=4, dstFilename=None):
        workers=30
        eval_every=10
        iterations=400
        passes=20

        self.util.logDebug('LDA', 'Training model...')
        self.model = LdaMulticore(self.corpus,workers=workers, num_topics=noOfTopics, id2word=self.dictionary,eval_every = None,iterations=iterations, passes=passes)
        self.util.logDebug('LDA', 'Model trained in ' + self.util.stopTimeTrack())
        print(self.model.print_topics())
        self.saveModel(dstFilename)
        self.loaded=True


    def saveModel(self, filename):
        self.util.logDebug('LDA', 'Saving model to ' + filename)
        self.model.save(filename)
        self.dictionary.save(filename+'.dict')
        MmCorpus.serialize(filename+'.corpus',self.corpus)
        self.util.logDebug('LDA', 'Saved in ' + self.util.stopTimeTrack())

    def loadModel(self, filename):
        self.util.logDebug('LDA', 'Loading model from ' + filename)
        self.model=LdaMulticore.load(fname=filename)
        self.dictionary=Dictionary.load(fname=filename+'.dict')
        self.corpus=MmCorpus(filename+'.corpus')
        print(self.dictionary)
        print(self.model.print_topic(0,topn=5))
        print(self.model.print_topic(1, topn=5))
        print(self.model.print_topic(2, topn=5))
        print(self.model.print_topic(3, topn=5))
        self.loaded=True
        self.util.logDebug('LDA', 'Model loaded in ' + self.util.stopTimeTrack())
        self.labelTopics(filename)

    def getTopTopic(self,inferenceOutput):
        thisDict=defaultdict(int)
        probList=[]
        for topic, prob in inferenceOutput:
            thisDict[str(prob)]=topic
            probList.append(prob)
        largestProb=max(probList)
        mostLikelyTopic=thisDict[str(largestProb)]
        return mostLikelyTopic

    def infer_topic_proba(self,string):
        import numpy as np
        prediction = [0.0,0.0,0.0,0.0]
        if (self.loaded):
            bow = self.dictionary.doc2bow(self.util.tokenize(string))
            results = self.model.get_document_topics(bow)
            for result in results:
                prediction[result[0]]=result[1]
        else:
            self.util.logError('LDA', 'Model is not loaded, cannot infer')
        prediction=np.array(prediction)
        return prediction

    def infer_topic(self, string):
        results=None
        if (self.loaded):
            bow=self.dictionary.doc2bow(self.util.tokenize(string))
            results=self.model.get_document_topics(bow)
        else:
            self.util.logError('LDA','Model is not loaded, cannot infer')
        results=self.getTopTopic(results)
        return results

    def visualizeLDA(self,filename):

        dictionary = Dictionary.load(filename+'.dict')
        corpus = MmCorpus(filename+'.corpus')
        lda = LdaMulticore.load(filename)
        self.util.logDebug('LDA', 'Preparing HTML ')
        ldavis=pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        self.util.logDebug('LDA', 'HTML prepared in ' + self.util.stopTimeTrack())
        pyLDAvis.save_html(ldavis, filename + '.html')
        self.util.logDebug('LDA', 'HTML saved in ' + self.util.stopTimeTrack())
#
# lda = LDA(logfilename='/home/kah1/test.log')
# lda.loadModel('/u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model')
# lda.labelTopics()
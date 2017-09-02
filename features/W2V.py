
"""
Vector model : Word2Vec
"""

from commons import Utilities
import glob
from gensim.models import Word2Vec
import gensim.utils
import numpy as np


class W2V():

    SENT_MODE_AVG='avg'
    SENT_MODE_S2V='s2v'
    corpus=None
    model=None
    util=None
    fitted=False

    def __init__(self, logFilename=None, utilObj=None, vectordim=100, window=5, wordFreqIgnored=1,epoches=200, noOfWorkers=10,learningRate=0.025):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)
        self.model = Word2Vec(size=vectordim, window=window, min_count=wordFreqIgnored, iter=epoches, workers=noOfWorkers,alpha=learningRate)
        self.util.logDebug('W2V', 'Initialising W2V')
        self.util.startTimeTrack()
        self.util.logDebug('W2V','Initialising W2V completed ' + self.util.stopTimeTrack())


    def saveVectorSpaceModel(self, dstFilename):
        self.util.logDebug('W2V', 'Saving model to ' + dstFilename)
        self.model.save(dstFilename)
        self.util.logDebug('W2V', 'Saving completed in ' + self.util.stopTimeTrack())
        pass


    def loadVectorSpaceModel(self, srcFilename):
        self.fitted=True
        self.util.logDebug('W2V', 'Loading model from ' + srcFilename)
        self.model=Word2Vec.load(srcFilename)
        self.util.logDebug('W2V','Loading completed in ' + self.util.stopTimeTrack())
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
        self.util.logDebug('W2V', 'Building and fitting corpus ')
        self.corpus=[]
        maxDocPerFolder=int(maxdocs/len(folderListOfCorpus.split(',')))
        docCounter=0
        for folder in folderListOfCorpus.split(','):
            self.util.logDebug('W2V', 'Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.*')):
                if (docCounter <= maxDocPerFolder):
                    fileContent=self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename),removeStopwords=True,toLowercase=True,replaceSlash=True,flatEmail=True,flatMonth=True,flatNumber=True,lemmatize=True), ' ')
                    # print(fileContent)
                    sentence=self.util.tokenize(gensim.utils.to_unicode(fileContent))
                    self.corpus.append(sentence)
                    docCounter=docCounter+1
                else:
                    docCounter=0
                    break

        self.util.logDebug('W2V', 'Corpus loaded in ' + self.util.checkpointTimeTrack())
        # self.corpus=[['sebastian', 'is', 'a', 'name'],['Jax', 'is' ,'here']]
        self.model.build_vocab(sentences=self.corpus)
        self.util.logDebug('W2V', 'Corpus built in ' + self.util.checkpointTimeTrack() + ' Training now...')
        self.model.train(sentences=self.corpus,total_examples=self.model.corpus_count,epochs=self.model.iter)
        self.fitted=True
        self.util.logDebug('W2V', 'Corpus fitted with ' + str((self.model.corpus_count)) + ' documents in ' + self.util.checkpointTimeTrack())
        self.saveVectorSpaceModel(dstFilename=dstFilename)
        self.util.logDebug('W2V', 'Model saved in ' + self.util.stopTimeTrack())
        # for item in self.model.wv.vocab.items():
        #     print('List vocab:',item)


    def inferVector(self, string=None, mode='avg'):
        """
        Since w2v only contains vectors for each word in the vocab, we will traverse through each word in the
        string according to our tokenisation. Get a vector of each word, and then average the vectors based on the
        words that are in the vocab.
        :param stringList:
        :return:
        """
        results=None
        if(self.fitted==False):
            self.util.logDebug('W2V', 'The corpus has yet to be loaded, this operation cannot proceed')
            exit(-1)
        else:
            if (mode==self.SENT_MODE_AVG):
                stringTokens=self.util.tokenize(rawStr=string)
                counter=0
                oldVector = None
                newVector = None

                for strToken in stringTokens:
                    try:
                        if (counter==0):
                            oldVector=self.model.wv[strToken]
                            counter=counter+1
                        else:
                            newVector=self.model.wv[strToken]
                            oldVector=np.vstack((oldVector,newVector))
                    except Exception as error:
                        # print (error)
                        # print(strToken+' not in vocab')
                        pass
                if (type(oldVector)!='NoneType'):
                    # print(oldVector.shape)
                    if (len(oldVector.shape) > 1):
                        results = np.average(oldVector, axis=0).reshape([1,self.model.vector_size])
                    else:
                        # There's only one vector(E.g. only 1 word in vocab)
                        results=oldVector.reshape([1,self.model.vector_size])

                else:
                    # No words found in vocab
                    # print('Vector size:' ,self.model.vector_size)
                    results=np.zeros(self.model.vector_size).reshape([1,self.model.vector_size])
            elif(mode==self.SENT_MODE_S2V):
                self.util.logError('W2V','Sentence2Vec is not implemented yet...exiting')
                exit(-1)
                pass
        return (np.array(results.tolist())).tolist()[0]



# w2v=W2V(logFilename='/u01/bigdata/02d_d2vModel1/log_cvW2vVectorSpaceModel_100.log')
# w2v.buildCorpus(folderListOfCorpus='/u01/bigdata/03a_01b_test/cvd2v/032/train', ngram=1, maxdocs=5, dstFilename='/u01/bigdata/02d_d2vModel1/cvW2vVectorSpaceModel_100.model', maxDim=100)
# vetor=w2v.inferVector('Non existing sebastian developing morning')
# vetor2=w2v.inferVector('morning')
# vetor3=w2v.inferVector('jax')
# print(vetor)
# print(vetor2)
# print(vetor3)
# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(vetor,vetor2))
# print(cosine_similarity(vetor,vetor3))
# print(cosine_similarity(vetor2,vetor3))
#
#
# w2v2=W2V(logFilename='/u01/bigdata/02d_d2vModel1/cvW2vVectorSpaceModel_100.model')
# w2v2.loadVectorSpaceModel('/u01/bigdata/02d_d2vModel1/cvW2vVectorSpaceModel_100.model')
# vetor=w2v2.inferVector('Non existing sebastian developing morning')
# vetor2=w2v2.inferVector('morning')
# vetor3=w2v2.inferVector('jax')
# print(vetor)
# print(vetor2)
# print(vetor3)
# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(vetor,vetor2))
# print(cosine_similarity(vetor,vetor3))
# print(cosine_similarity(vetor2,vetor3))
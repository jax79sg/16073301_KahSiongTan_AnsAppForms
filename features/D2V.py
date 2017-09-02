"""
Vector model Document2Vec
"""
from commons import Utilities
import glob
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import gensim.utils

class D2V():

    corpus=None
    model=None
    util=None
    fitted=False

    def __init__(self, logFilename=None, utilObj=None, vectordim=100, wordFreqIgnored=1,epoches=200, noOfWorkers=10,learningRate=0.025):
        if (utilObj!=None):
            self.util=utilObj
        elif(logFilename!=None):
            self.util = Utilities.Utility()
            self.util.setupLogFileLoc(logFilename)
        self.model = Doc2Vec(size=vectordim, min_count=wordFreqIgnored, iter=epoches, workers=noOfWorkers,alpha=learningRate)
        self.util.logDebug('D2V', 'Initialising D2V')
        self.util.startTimeTrack()
        self.util.logDebug('D2V','Initialising D2V completed ' + self.util.stopTimeTrack())


    def saveVectorSpaceModel(self, dstFilename):
        self.util.logDebug('D2V', 'Saving model to ' + dstFilename)
        self.model.save(dstFilename)
        self.util.logDebug('D2V', 'Saving completed in ' + self.util.stopTimeTrack())
        pass


    def loadVectorSpaceModel(self, srcFilename):
        self.fitted=True
        self.util.logDebug('D2V', 'Loading model from ' + srcFilename)
        self.model=Doc2Vec.load(srcFilename)
        self.util.logDebug('D2V','Loading completed in ' + self.util.stopTimeTrack())
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
        self.util.logDebug('D2V', 'Building and fitting corpus ')
        self.corpus=[]
        maxDocPerFolder=int(maxdocs/len(folderListOfCorpus.split(',')))
        docCounter=0
        for folder in folderListOfCorpus.split(','):
            self.util.logDebug('D2V', 'Processing ' + folder)
            for filename in sorted(glob.iglob(folder + '/*.*')):
                if (docCounter <= maxDocPerFolder):
                    fileContent=self.util.tokensToStr(self.util.tokenize(self.util.readFileContent(filename=filename),removeStopwords=True,toLowercase=True,replaceSlash=True,flatEmail=True,flatMonth=True,flatNumber=True,lemmatize=True), ' ')
                    # print(fileContent)
                    tag=docCounter
                    sentence=self.util.tokenize(gensim.utils.to_unicode(fileContent))
                    td=TaggedDocument(sentence,[tag])
                    self.corpus.append(td)
                    docCounter=docCounter+1
                else:
                    docCounter=0
                    break

        self.util.logDebug('D2V', 'Corpus loaded in ' + self.util.stopTimeTrack())
        # self.corpus=[['sebastian', 'is', 'a', 'name'],['Jax', 'is' ,'here']]
        self.model.build_vocab(sentences=self.corpus)
        self.model.train(sentences=self.corpus,total_examples=self.model.corpus_count,epochs=self.model.iter)
        self.fitted=True
        self.util.logDebug('D2V', 'Corpus fitted with ' + str((self.model.corpus_count)) + ' documents in ' + self.util.stopTimeTrack())
        self.saveVectorSpaceModel(dstFilename=dstFilename)
        self.util.logDebug('D2V', 'Model saved in ' + self.util.stopTimeTrack())

    def inferVector(self, string=None):
        """
        Since w2v only contains vectors for each word in the vocab, we will traverse through each word in the
        string according to our tokenisation. Get a vector of each word, and then average the vectors based on the
        words that are in the vocab.
        :param stringList:
        :return:
        """
        results=None
        if(self.fitted==False):
            self.util.logDebug('D2V', 'The corpus has yet to be loaded, this operation cannot proceed')
            exit(-1)
        else:
            sentenceTokens = self.util.tokenize(rawStr=string)
            results = self.model.infer_vector(sentenceTokens)
            results=results.reshape([1,results.shape[0]])
        return results.tolist()[0]

#
# d2v=D2V(logFilename='/u01/bigdata/02d_d2vModel1/log_cvD2vVectorSpaceModel_100.log')
# d2v.buildCorpus(folderListOfCorpus='/u01/bigdata/03a_01b_test/cvd2v/032/train', ngram=1, maxdocs=5, dstFilename='/u01/bigdata/02d_d2vModel1/cvD2vVectorSpaceModel_100.model', maxDim=100)
# vetor=d2v.inferVector('Non existing sebastian developing morning')
# vetor2=d2v.inferVector('morning')
# vetor3=d2v.inferVector('jax')
# print(vetor)
# print(vetor2)
# print(vetor3)
# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(vetor,vetor2))
# print(cosine_similarity(vetor,vetor3))
# print(cosine_similarity(vetor2,vetor3))
#
#
# d2v2=D2V(logFilename='/u01/bigdata/02d_d2vModel1/cvD2vVectorSpaceModel_100.model')
# d2v2.loadVectorSpaceModel('/u01/bigdata/02d_d2vModel1/cvD2vVectorSpaceModel_100.model')
# vetor=d2v2.inferVector('Non existing sebastian developing morning')
# vetor2=d2v2.inferVector('morning')
# vetor3=d2v2.inferVector('jax')
# print(vetor)
# print(vetor2)
# print(vetor3)
# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(vetor,vetor2))
# print(cosine_similarity(vetor,vetor3))
# print(cosine_similarity(vetor2,vetor3))
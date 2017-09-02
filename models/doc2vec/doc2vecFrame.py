"""
DOC2VEC
This class performs doc2vec training based on provided d2v files.
"""
import sys
sys.path.append("/home/kah1/remote_cookie_runtime")
sys.path.append("/home/kah1/remote_cookie_runtime/src")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromAppJSON")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromCV")
sys.path.append("/home/kah1/remote_cookie_runtime/src/models/doc2vec")
from gensim.models import Doc2Vec
import models.doc2vec.ParagraphLoaders as load
import pandas as pd
from commons import Utilities

class doc2vecFrame():
    __model=None
    __documents=None
    __util=None
    globallogfile=None

    def __init__(self, vectordim=100, wordFreqIgnored=2,epoches=20, noOfWorkers=20,learningRate=0.025, distributedMem=1, logFilename=None, utilObj=None):
        if (utilObj!=None):
            self.__util=utilObj
        elif(logFilename!=None):
            self.__util = Utilities.Utility()
            self.__util.setupLogFileLoc(logFilename)
        self.__util.logDebug(self.__util.DOC2VECFRAME, 'Initialising doc2vec')
        self.__util.startTimeTrack()
        # self.__model = Doc2Vec(hashfxn=customHash.customHash, size=vectordim, min_count=wordFreqIgnored, iter=epoches, workers=noOfWorkers, alpha=learningRate, dm=distributedMem)
        self.__model = Doc2Vec(size=vectordim, min_count=wordFreqIgnored, iter=epoches, workers=noOfWorkers, alpha=learningRate, dm=distributedMem)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Initialising doc2vec completed ' + self.__util.stopTimeTrack())
        #self.__model = Doc2Vec(size=5,alpha=0.025,min_alpha=0.025,min_count=1,dm=0)
        # print(FAST_VERSION)
        # os.environ['PYTHONHASHSEED']='0' #This is required to be able to reproduce same results.

    def getNumpVectors(self):
        return self.__model.docvecs.doctag_syn0

    def getAllDocVecs(self):
        """
        Generate a dataframe containing the document tag and its relevant vector
        :return:
        """
        doctagvec = pd.DataFrame(columns=('doctag', 'docvec'))
        self.__util.checkpointTimeTrack()
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Building docvec dataframe ')
        for item in self.__model.docvecs.doctags.items():
            doctag=item[0]
            docvec=self.__model.docvecs[item[0]]
            newrow = pd.DataFrame(data={'doctag': [doctag], 'docvec': [docvec]})
            doctagvec = doctagvec.append(newrow)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Building docvec dataframe completed in ' + self.__util.stopTimeTrack())
        return doctagvec

    def saveAllDocVecsToCSV(self, csvFile=None):
        doctagvecDF=self.getAllDocVecs()
        doctagvecDF.to_csv(csvFile, ',', mode='w',header=True, index=False, columns=('doctag', 'docvec'))


    def __buildVocabw2v(self, listOfTaggedDocuments):

        self.__util.logDebug(self.__util.DOC2VECFRAME, 'Building vocab for ' + str(len(listOfTaggedDocuments)) + ' documents')
        self.__util.startTimeTrack()
        self.__model.build_vocab(listOfTaggedDocuments)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Completed vocab building in ' + self.__util.stopTimeTrack())


    def train(self, strFolderPathToDocs, modelSaveFilenamePath):
        """

        :param strFolderPathToDocs: This will look for .d2v files as a document.
        :param modelSaveFilenamePath: Path to save the model.
        :return: None
        """
        documents = load.get_doc(folder_name=strFolderPathToDocs,util=self.__util)
        #print(len(documents), type(documents))
        self.__buildVocabw2v(documents)

        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Training doc2v ...')
        self.__util.startTimeTrack()
        self.__model.train(documents, total_words=self.__model.corpus_count, epochs=self.__model.iter)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Completed training in ' + self.__util.stopTimeTrack())

        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Saving doc2v model...')
        self.__util.startTimeTrack()
        self.__model.save(modelSaveFilenamePath)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Completed saving in ' + self.__util.stopTimeTrack())

        documents=None

    def inferVector(self,str=None):
        sentenceTokens=self.__util.tokenize(rawStr=str)
        docVector=self.__model.infer_vector(sentenceTokens)
        return docVector

    def trainFromMultiFolders(self, strFolderPathToDocsList, modelSaveFilenamePath, maxDocsToTrain=-1):
        """
        Using a list of folders instead of just one folder
        :param strFolderPathToDocsList: This will look for .d2v files as a document.
        :param modelSaveFilenamePath: Path to save the model.
        :return: None
        """
        documents=[]
        for folderName in strFolderPathToDocsList.split(','):
            documentsFromFolder = load.get_doc(folder_name=folderName, util=self.__util, maxdocs=maxDocsToTrain)
            documents=documents +documentsFromFolder

        #print(len(documents), type(documents))
        self.__buildVocabw2v(documents)

        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Training doc2v ...')
        self.__util.startTimeTrack()
        self.__model.train(documents, total_words=self.__model.corpus_count, epochs=self.__model.iter)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Completed training in ' + self.__util.stopTimeTrack())

        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Saving doc2v model...')
        self.__util.startTimeTrack()
        self.__model.save(modelSaveFilenamePath)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Completed saving in ' + self.__util.stopTimeTrack())

        documents=None

    def sim2tags(self,tag1,tag2):
        self.__model.docvecs.similarity(tag1,tag2)

    def documentsSimilarToDoc(self,doc,topn):
        """
        Given a doc,retrieve similar docs in decreasing similarity order.
        :param doc: A document of text. tokenised in a list (E.g. ['This','is','a','python','script']
        :param topn: Determines the top Nth similar documents to return
        :return: Returns a list of ('tagid',semSimScore) with reference to doc. Where tagid is the id tagged to the doc in train model
        """
        # self.__model.random.seed(0)
        # print('Runtime:',self.__model.infer_vector(doc))
        # self.__model.random.seed(0)
        # print('Runtime:',self.__model.infer_vector(doc))

        # self.__model.random.seed(0)
        docList=self.__model.docvecs.most_similar([self.__model.infer_vector(doc)], topn=topn)
        return docList

    def loadModel(self,modelFilenamePath):
        """
        Load trained model.
        :param modelFilenamePath:
        :return:
        """
        self.__util.logDebug(self.__util.DOC2VECFRAME, 'Loading model from ' + modelFilenamePath)
        self.__util.startTimeTrack()
        self.__model=Doc2Vec.load(modelFilenamePath)
        self.__util.logDebug(self.__util.DOC2VECFRAME,
                             'Loading completed in ' + self.__util.stopTimeTrack())

    def getAvgSimilarity(self, tokens=None, averageCount=3):
        """
        For this set of tokens, infer a vector and then look for 3 most similar documents in the vector space
        Average the sim score and then return the score
        :param tokens:
        :return:
        """
        top3results = self.documentsSimilarToDoc(tokens, averageCount)
        total = 0
        resultcounter = 0
        for result in top3results:
            score = result[1]
            total = total + score
            resultcounter = resultcounter + 1
        avgResult = total / resultcounter
        return avgResult

sys.path.append("/home/kah1/remote_cookie_runtime")
sys.path.append("/home/kah1/remote_cookie_runtime/python")
sys.path.append("/home/kah1/remote_cookie_runtime/python/extractFromAppJSON")
sys.path.append("/home/kah1/remote_cookie_runtime/python/extractFromCV")
sys.path.append("/home/kah1/remote_cookie_runtime/python/doc2vec")

# Change the content here as required.
# python3 doc2vecFrame.py 'train' '/u01/bigdata/02c_d2vModel1/log_train32Edu.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train'
# python3 doc2vecFrame.py 'train' '/u01/bigdata/02c_d2vModel1/log_train32Skills.log' '/u01/bigdata/02c_d2vModel1/train32Skills.model' '/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train'
# python3 doc2vecFrame.py 'train' '/u01/bigdata/02c_d2vModel1/log_train32PersonalDetails.log' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model' '/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train'

# python3 doc2vecFrame.py 'load' '/u01/bigdata/02c_d2vModel1/log_train32Edu.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model'
# python3 doc2vecFrame.py 'load' '/u01/bigdata/02c_d2vModel1/log_train32Skills.log' '/u01/bigdata/02c_d2vModel1/train32Skills.model'
# python3 doc2vecFrame.py 'load' '/u01/bigdata/02c_d2vModel1/log_train32PersonalDetails.log' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model'

# if __name__ == "__main__":
#     print('Usage: python3 doc2vecFrame.py train|load logfilename [modeFilename] [src folder]')
#     if(len(sys.argv)>2):
#         operation = sys.argv[1]
#         logFile=sys.argv[2]
#         modelFile=None
#         srcFolder=None
#         print('Logging to ', logFile)
#         doc2vecframe = doc2vecFrame(logFilename=logFile)
#
#         if(operation=='train'):
#             if len(sys.argv)==5:
#                 modelFile = sys.argv[3]
#                 srcFolder = sys.argv[4]
#                 print('operation: ', operation)
#                 print('modelFile: ', modelFile)
#                 print('srcFolder: ', srcFolder)
#                 doc2vecframe.train(srcFolder,modelFile)
#             else:
#                 print("Train operations requires following arguments\n[\'train\'|\'load\' [model filename] [source folder]")
#                 sys.exit(1)
#         elif(operation=='load'):
#             if (len(sys.argv)>2):
#                 modelFile = sys.argv[3]
#                 print('operation: ', operation)
#                 print('modelFile: ', modelFile)
#                 doc2vecframe.loadModel(modelFile)
#             else:
#                 print("Load operations requires following arguments\n[\'train\'|\'load\' [model filename]")
#                 sys.exit(1)
#     else:
#         print("No arguments provided..")
#         exit(-1)
#
#     top3results = doc2vecframe.documentsSimilarToDoc(
#         ['German', 'English', 'Spanish', 'Mandarin', 'Latin','German','Latinum', 'Chinese','Japanese','Korean','Thai','Hokkien','Dialect'], 3)
#     total = 0
#     resultcounter = 0
#     for result in top3results:
#         score = result[1]
#         total = total + score
#         resultcounter = resultcounter + 1
#     avgResult = total / resultcounter
#     print('German: ' + str(avgResult))
#
#     top3results = doc2vecframe.documentsSimilarToDoc(
#         ['Computer','IT','skills','C++','Proficiency','Intermediate','Computer','IT','skills','Java','Java','Servlets','JSP','Proficiency','Intermediate','Computer','IT','Skills','Microsoft','Excel','Proficiency','Intermediate','Computer','IT','skills','Microsoft',' Poewrpoint'], 3)
#     total = 0
#     resultcounter = 0
#     for result in top3results:
#         score = result[1]
#         total = total + score
#         resultcounter = resultcounter + 1
#     avgResult = total / resultcounter
#     print('Powerpoint: ' + str(avgResult))
#
#     top3results = doc2vecframe.documentsSimilarToDoc(
#         ['Bullshit','and'], 3)
#     total = 0
#     resultcounter = 0
#     for result in top3results:
#         score = result[1]
#         total = total + score
#         resultcounter = resultcounter + 1
#     avgResult = total / resultcounter
#     print('Bullshit: ' + str(avgResult))
#
#     top3results = doc2vecframe.documentsSimilarToDoc(
#         ['Microsoft','powerpoint','word','excel','java','serlvet','and','javascipt','computer','hardware','and','software'], 3)
#     total = 0
#     resultcounter = 0
#     for result in top3results:
#         score = result[1]
#         total = total + score
#         resultcounter = resultcounter + 1
#     avgResult = total / resultcounter
#     print('My own computer describe: ' + str(avgResult))

# top3results = doc2vecframe.documentsSimilarToDoc(
#     ['Jax','Tan','email','jax79sg@yahoo.com.sg','tel','65621507','Address','Blk','244','Jurong','East','St','24','Singapore','600244'], 3)
# total = 0
# resultcounter = 0
# for result in top3results:
#     score = result[1]
#     total = total + score
#     resultcounter = resultcounter + 1
# avgResult = total / resultcounter
# print('My personal describe: ' + str(avgResult))
#
#
# top3results = doc2vecframe.documentsSimilarToDoc(
#     ['student'], 3)
# total = 0
# resultcounter = 0
# for result in top3results:
#     score = result[1]
#     total = total + score
#     resultcounter = resultcounter + 1
# avgResult = total / resultcounter
# print('Sample personal details: ' + str(avgResult))





# doc2vecframe.train('/home/kah1/remote_gitlab_source/datasamples/testdocfolder','/home/kah1/remote_gitlab_source/datasamples/testdocfolder/doc2vec.model')
# doc2vecframe.train('/u01/bigdata/d2v_QandA','/u01/bigdata/d2vModel2/doc2vec.model')
# doc2vecframe.loadModel('/u01/bigdata/d2vModel2/doc2vec.model')

#Print twice to confirm reproducibility.
# print(doc2vecframe.documentsSimilarToDoc(['Degree','in','Philosophy','from','Thinking','University','from','2009','to','2010'],3))
# print(doc2vecframe.documentsSimilarToDoc(['Degree','in','Philosophy','from','Thinking','University','from','2009','to','2010'],3))
# print(doc2vecframe.documentsSimilarToDoc(['German','English', 'Spanish', 'Mandarin', 'Latin (German Latinum)'],3))
# print(doc2vecframe.documentsSimilarToDoc(['Experienced','in','using','Microsoft','Word','and','Powerpoint'],3))


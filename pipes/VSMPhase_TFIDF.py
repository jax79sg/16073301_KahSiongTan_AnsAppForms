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
from features.TFIDF import TFIDF

# Change the content here as required.
# python3 /home/kah1/remote_cookie_runtime/src/pipes/featurePhase3_TFIDF.py '/u01/bigdata/02d_d2vModel1/log_cvTfidf10000VectorSpaceModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_10000.model' '1' '10000000' '10000'
# python3 /home/kah1/remote_cookie_runtime/src/pipes/featurePhase3_TFIDF.py '/u01/bigdata/02d_d2vModel1/log_cvTfidf5000VectorSpaceModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '/u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_5000.model' '1' '10000000' '5000'

if __name__ == "__main__":
    if(len(sys.argv)==8):
        logFile = sys.argv[1]
        folderOfDocuments=sys.argv[2]
        dstVectorSpaceModelFilename=sys.argv[3]
        ngram=int(sys.argv[4])
        maxSize = int(sys.argv[5])
        maxDim = int(sys.argv[6])
        tokenRules = (sys.argv[7])


        print('logFile:',logFile)
        print('folderOfDocuments:', folderOfDocuments)
        print('dstVectorSpaceModelFilename:', dstVectorSpaceModelFilename)
        print('ngram:', ngram)
        print('maxSize:', maxSize)
        print('maxDim:', maxDim)
        print('tokenRules',tokenRules)

        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        util.setupTokenizationRules(tokenRules)

        bow=TFIDF(utilObj=util, maxDim=maxDim,ngram=ngram)
        bow.buildCorpus(folderListOfCorpus=folderOfDocuments, ngram=ngram, maxdocs=maxSize, dstFilename=dstVectorSpaceModelFilename)

        vector=bow.inferVector(util.tokensToStr(util.tokenize('This is the skill that is from IT. Microsoft Office, Word, Powerpoint, Excel')))

        bow2 = TFIDF(utilObj=util, maxDim=maxDim)
        bow2.loadVectorSpaceModel(dstVectorSpaceModelFilename)
        vector2=bow2.inferVector(util.tokensToStr(util.tokenize('This is the skill that is from IT. Microsoft Office, Word, Powerpoint, Excel')))

        # util.logInfo('Vector1Size:', str(vector.toarray().shape[0]))
        # util.logInfo('Vector1Size:', str(vector.toarray().shape[1]))

        # print('Vector1:', vector.toarray().shape)
        # print('Vector2:', vector2.toarray().shape)
        if (util.compareCsrMatrix(vector,vector2)):
            print('Vectors are likely to be the same!')
        else:
            print('Vectors are NOT the same!')

    else:
        print("No arguments provided..")
        exit(-1)
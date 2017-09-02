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
from features.LDA import LDA

# Change the content here as required.
# python3 /home/kah1/remote_cookie_runtime/src/pipes/VSMPhase_LDA.py '/u01/bigdata/02d_d2vModel1/log_CvLda4TopicModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '100000000' '/u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model' 4
# python3 /home/kah1/remote_cookie_runtime/src/pipes/VSMPhase_LDA.py '/u01/bigdata/02d_d2vModel1/log_CvLda5TopicModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '100000000' '/u01/bigdata/02d_d2vModel1/CvLda5TopicModel.model' 5

if __name__ == "__main__":
    if(len(sys.argv)==7):
        logFile = sys.argv[1]
        folderListOfCorpus=sys.argv[2]
        maxdocs=int(sys.argv[3])
        dstFilename=(sys.argv[4])
        noOfTopics = int(sys.argv[5])
        tokenRules = (sys.argv[6])

        print('logFile:',logFile)
        print('folderListOfCorpus:', folderListOfCorpus)
        print('maxdocs:', maxdocs)
        print('dstFilename:', dstFilename)
        print('noOfTopics:', noOfTopics)
        print('tokenRules:', tokenRules)

        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        util.setupTokenizationRules(tokenRules)

        lda = LDA(utilObj=util)
        lda.buildCorpus(folderListOfCorpus=folderListOfCorpus, maxdocs=maxdocs)
        lda.trainModel(noOfTopics=noOfTopics,dstFilename=dstFilename)
        lda.visualizeLDA(dstFilename)
    else:
        print("No arguments provided..")
        exit(-1)
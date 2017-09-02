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
sys.path.append("/home/kah1/cookie/ksthesis")
sys.path.append("/home/kah1/cookie/ksthesis/src")
sys.path.append("/home/kah1/cookie/ksthesis/src/commons")
from models.SigificanceTests.CompileSignificanceResults import CompileSignificanceResults
import glob
from commons import Utilities
import itertools

# python3 /home/kah1/remote_cookie_runtime/src/pipes/ModelPhase_GenerateSignificantTest.py '/u01/bigdata/02d_d2vModel1/features/log_generateSignificance.log' 'stuart,wilcoxon' '/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/significance/stuart,/u01/bigdata/02d_d2vModel1/featureset3NoPreproc/significance/wilcoxon' 'LOG_BOW5000,LOG_D2V100,LOG_TFIDF5000,LOG_W2V100min1,MLP_BOW5000,MLP_D2V100,MLP_TFIDF5000,MLP_W2V100min1,NB_BOW5000,NBG_D2V100,NBG_W2V100min1,NB_TFIDF5000,XGB_BOW5000,XGB_D2V100,XGB_TFIDF5000,XGB_W2V100min1,SIM_BOW5000,SIM_D2V100,SIM_TFIDF5000,SIM_W2V100min1,SVM_BOW5000,SVM_D2V100,SVM_TFIDF5000,SVM_W2V100min1,TOPIC_NONE' '/u01/bigdata/02d_d2vModel1/features/significanceResults.csv'

if __name__ == "__main__":
    if(len(sys.argv)==6):
        logFile = sys.argv[1]
        typeStr=sys.argv[2]
        folderStr=sys.argv[3]
        targetMethodsStr = sys.argv[4]
        saveResults=sys.argv[5]

        print('logFile:',logFile)
        print('typeStr:', typeStr)
        print('folderStr:', folderStr)
        print('targetMethodsStr:', targetMethodsStr)
        print('saveResults:', saveResults)
        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)

        sigTest = CompileSignificanceResults(utilObj=util)
        typeList=typeStr.split(',')
        folderList=folderStr.split(',')
        targetMethodsList=targetMethodsStr.split(',')

        for index in range(0,len(typeList)):
            type=typeList[index]
            folder=folderList[index]
            sigTest.addTest(type=type,folderOfResults=folder)

        sigTest.createSigTable(listOfApproaches=targetMethodsList, saveFilename=saveResults)
        pass

    else:
        print("No arguments provided..")
        exit(-1)
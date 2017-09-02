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
from commons import Utilities
from models.topicClusterModel import TopicClusteringPredictionModel
from models import Evaluator
import glob
#Cross Val

if __name__ == "__main__":
    if(len(sys.argv)==6 or len(sys.argv)==9):
        logFile = sys.argv[1]
        ldaModelFilename=sys.argv[2]
        sampleLabelledTestFilename=sys.argv[3]
        approach_vsm = sys.argv[4]
        tokenRules = sys.argv[5]
        print('logFile:',logFile)
        print('ldaModelFilename:', ldaModelFilename)
        print('sampleLabelledTestFilename:', sampleLabelledTestFilename)
        print('approach_vsm:', approach_vsm)
        print('tokenRules:', tokenRules)

        listOfAppD2vFoldersOrderByLabel=None
        listOfLabelsOrderByLabel=None
        appFolderLocation=None
        if(len(sys.argv)==9):
            listOfAppD2vFoldersOrderByLabelStr = sys.argv[6]
            listOfAppD2vFoldersOrderByLabel=listOfAppD2vFoldersOrderByLabelStr.split(',')
            listOfLabelsOrderByLabelStr = sys.argv[7]
            listOfLabelsOrderByLabel=list(map(int,listOfLabelsOrderByLabelStr.split(',')))
            appFolderLocation = sys.argv[8]

            print('listOfAppD2vFoldersOrderByLabelStr:', listOfAppD2vFoldersOrderByLabelStr)
            print('listOfLabelsOrderByLabelStr:', listOfLabelsOrderByLabelStr)
            print('appFolderLocation:', appFolderLocation)


        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        predictionModel=TopicClusteringPredictionModel.TopicClusteringPredictionModel(utilObj=util,ldaModelFilename=ldaModelFilename)

        # Loop to load all CV folds
        resultsCvF1 = []

        counter = 0
        for splittedFeature in (glob.iglob(sampleLabelledTestFilename + '.0*')):
            counter = counter + 1

        for i in range(1, counter):
            newCVFilename = sampleLabelledTestFilename.split('.')[0] + '.features.' + str(i).zfill(3)
            if (util.ifFileExists(newCVFilename)==False):
                util.logError('CrossVal',newCVFilename + ' not found. Did you forget to run features/split_features.py? Exiting')
                exit(-1)

            predictionModel.flushTestData()
            predictionModel.loadXYtest(newCVFilename)
            eval = Evaluator.Evaluator(utilObj=util)
            resultsAcc, resultsPrec, resultsRecall, resultsF1, resultsPrecperclass, resultsRecallperclass, resultsF1perclass = predictionModel.evaluate(
                approach_vsm+str(i).zfill(3), eval)
            resultsCvF1.append(resultsF1)

        util.saveListToFile(resultsCvF1, approach_vsm + '.CROSSVAL.F1')

    else:
        print("No arguments provided..")
        exit(-1)
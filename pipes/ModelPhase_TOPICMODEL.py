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
from models import HeuristicEvaluator


# Change the content here as required.
# python3 /home/kah1/remote_cookie_runtime/src/pipes/ModelPhase_TOPICMODEL.py '/u01/bigdata/02d_d2vModel1/log_CvLda4TopicModel.eval' '/u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v100min1.features' 'TOPIC_NONE' 'education,skill|language,personal,experience' '0,1,2,3' '/u01/bigdata/00_appjson'

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
        util.setupTokenizationRules(tokenRules)
        predictionModel=TopicClusteringPredictionModel.TopicClusteringPredictionModel(utilObj=util,ldaModelFilename=ldaModelFilename)
        predictionModel.loadXYtest(sampleLabelledTestFilename)
        eval = Evaluator.Evaluator(utilObj=util)
        predictionModel.evaluate(approach_vsm, eval)

        if (len(sys.argv) == 9):
            evalHeu = HeuristicEvaluator.HeuristicEvaluator(utilObj=util, listOfAppD2vFoldersOrderByLabel=listOfAppD2vFoldersOrderByLabel, listOfLabelsOrderByLabel=listOfLabelsOrderByLabel,appFolderLocation=appFolderLocation)
            predictionModel.evaluate(approach_vsm, evalHeu)

    else:
        print("No arguments provided..")
        exit(-1)
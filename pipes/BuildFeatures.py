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

from features.build_features import FeatureBuilder
from commons import Utilities
#python3 /home/kah1/remote_cookie_runtime/src/pipes/BuildFeatures.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainBOW10000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_10000.model' '/u01/bigdata/02d_d2vModel1/features/appd2vTrainBOW10000.features'
#python3 /home/kah1/remote_cookie_runtime/src/pipes/BuildFeatures.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainBOW5000features.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train,/u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train,/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/train,/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/train' '0,1,2,3' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/appd2vTrainBOW5000.features'
#python3 /home/kah1/remote_cookie_runtime/src/pipes/BuildFeatures.py '/u01/bigdata/02d_d2vModel1/features/log_appd2vTrainBOW5000features.log' '' 'None' 'bow' '/u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_5000.model' '/u01/bigdata/02d_d2vModel1/features/cvd2vTrainBOW5000.features'
if(len(sys.argv)==8):
    logLoc = sys.argv[1]
    srcFolderList = sys.argv[2]
    labelList = sys.argv[3]
    vsmType = sys.argv[4]
    vsmModelFilename = sys.argv[5]
    dstFilename = sys.argv[6]
    tokenRules= sys.argv[7]

    print('logloc: ', logLoc)
    print('srcFolderList: ',srcFolderList)
    print('labelList: ', labelList)
    print('vsmType: ', vsmType)
    print('vsmModelFilename: ', vsmModelFilename)
    print('dstFilename: ', dstFilename)
    print('tokenRules: ', tokenRules)

    util = Utilities.Utility()
    util.setupLogFileLoc(logFile=logLoc)
    util.setupTokenizationRules(tokenRules)

    fbuild = FeatureBuilder(utilObj=util)
    fbuild.buildFeatures(srcFolderList=srcFolderList, labelList=labelList, vsmType=vsmType, vsmModelFilename=vsmModelFilename, dstFilename=dstFilename)

else:
    print("No arguments provided..")
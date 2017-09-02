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
from data.extractFromCV.CVtoSections import CVtoSections
from features.build_features import FeatureBuilder
# Change the content here as required.
#python3 BatchCreateFeaturesFromCVs.py '/u01/bigdata/02d_d2vModel1/features/log_createClient200FeaturesFromCVsW2V300min1.log' '/u01/bigdata/00_cv_text/CV2/200/temp' '/u01/bigdata/03a_01b_test/cvd2v/200/test' 'w2v' '/u01/bigdata/02d_d2vModel1/featureset2/cvW2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient200TrainW2v300min1.features'
#python3 BatchCreateFeaturesFromCVs.py '/u01/bigdata/02d_d2vModel1/features/log_createClient200FeaturesFromCVsW2V300min2.log' '/u01/bigdata/00_cv_text/CV2/200/temp' '/u01/bigdata/03a_01b_test/cvd2v/200/test' 'w2v' '/u01/bigdata/02d_d2vModel1/featureset2/cvW2v300_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient200TrainW2v300min2.features'
#python3 BatchCreateFeaturesFromCVs.py '/u01/bigdata/02d_d2vModel1/features/log_createClient200FeaturesFromCVsW2V100min1.log' '/u01/bigdata/00_cv_text/CV2/200/temp' '/u01/bigdata/03a_01b_test/cvd2v/200/test' 'w2v' '/u01/bigdata/02d_d2vModel1/featureset2/cvW2v100VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient200TrainW2v100min1.features'
#python3 BatchCreateFeaturesFromCVs.py '/u01/bigdata/02d_d2vModel1/features/log_createClient200FeaturesFromCVsW2V100min1.log' '/u01/bigdata/00_cv_text/CV2/200/temp' '/u01/bigdata/03a_01b_test/cvd2v/200/test' 'w2v' '/u01/bigdata/02d_d2vModel1/featureset2/cvW2v100_min2VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient200TrainW2v100min2.features'
"""
Given srcCVFolder, destCVSectionFolder (Make sure this is empty), maxNoOfCVs.
- Traverse srcCVFolder for txt files (Converted CVs)
- For each txt file
    - Break into CV sections (cvd2v files)
    - Save into destCVSectionFolder
    - break when maxNoOfCVs reached
- Traverse destCVSectionFolder for cvd2v files (CV sections)
    -  Build the features file based on this content.
    

"""

if __name__ == "__main__":
    if(len(sys.argv)==8):#python3 BatchCreateFeaturesFromCVs.py '/u01/bigdata/02d_d2vModel1/features/log_createClient291FromCVsW2V100min1.log' '/u01/bigdata/00_cv_text/CV2/291/temp' '/u01/bigdata/03a_01b_test/cvd2v/291/test' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v100min1VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvClient291TrainW2v100min1.features' 'removeStopwords,toLowercase,replaceSlash'
        logFile = sys.argv[1]
        srcCVfolder=sys.argv[2]
        destCVSectionsFolder=sys.argv[3]
        vsmType=sys.argv[4]
        vsmModelFilename=sys.argv[5]
        dstFilename=sys.argv[6]
        tokenRules=sys.argv[7]

        print('logFile:',logFile)
        print('srcCVfolder:', srcCVfolder)
        print('destCVSectionsFolder:', destCVSectionsFolder)
        print('vsmType:', vsmType)
        print('vsmModelFilename:', vsmModelFilename)
        print('dstFilename:', dstFilename)
        print('tokenRules:', tokenRules)

        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        util.setupTokenizationRules(tokenRules)


        if (util.isFolderEmpty(destCVSectionsFolder)==False):
            util.logError('BatchCreateFeaturesFromCVs',destCVSectionsFolder + ' is not empty. Whatever is in there will be included in features')
            util.recreateDir(destCVSectionsFolder)
        cvExtractor=CVtoSections(utilObj=util)
        cvExtractor.extractCVunderFolder(cvFolder=srcCVfolder,folderToStoreD2V=destCVSectionsFolder)
        fb=FeatureBuilder(utilObj=util)
        fb.buildFeatures(srcFolderList=destCVSectionsFolder,labelList='-1',vsmType=vsmType,vsmModelFilename=vsmModelFilename,dstFilename=dstFilename)

    else:
        print("No arguments provided..")
        exit(-1)
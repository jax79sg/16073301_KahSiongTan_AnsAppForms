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
from data.extractFromCV.CVtoSections import CVtoSections

#python3 CVtoSections.py '/u01/bigdata/03a_01b_test/log_extractCVd2vTrainset.log' '/u01/bigdata/01b_d2v/032/edu/trainset.csv' '/u01/bigdata/01b_d2v/032/skills/trainset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/trainset.csv' '/u01/bigdata/01b_d2v/032/workexp/trainset.csv' '/u01/bigdata/03a_01b_test/cvd2v/032/train'
#python3 CVtoSections.py '/u01/bigdata/03a_01b_test/log_extractCVd2vTestset.log' '/u01/bigdata/01b_d2v/032/edu/testset.csv' '/u01/bigdata/01b_d2v/032/skills/testset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/testset.csv' '/u01/bigdata/01b_d2v/032/workexp/testset.csv' '/u01/bigdata/03a_01b_test/cvd2v/032/test'
if(len(sys.argv)==7):
    logLoc = sys.argv[1]
    datasetEduFilename = sys.argv[2]
    datasetSkillsFilename = sys.argv[3]
    datasetPersonalDetailsFilename = sys.argv[4]
    datasetWorkexpFilename = sys.argv[5]
    folderToStoreD2V = sys.argv[6]
    print('logloc: ', logLoc)
    print('datasetEduFilename: ',datasetEduFilename)
    print('datasetSkillsFilename: ', datasetSkillsFilename)
    print('datasetPersonalDetailsFilename: ', datasetPersonalDetailsFilename)
    print('datasetWorkexpFilename: ', datasetWorkexpFilename)
    print('folderToStoreD2V: ', folderToStoreD2V)

    if(folderToStoreD2V=='' or logLoc==''):
        print("No arguments provided..")
    else:

        cvExtractor = CVtoSections(logLoc)
        # cvExtractor.extractCVunderFolder(cvFolder=srcLoc, folderToStoreD2V=d2vLoc)
        cvExtractor.extractCVunderDataset(datasetFilename=[datasetEduFilename,datasetEduFilename,datasetPersonalDetailsFilename,datasetWorkexpFilename], folderToStoreD2V=folderToStoreD2V)
else:
    print("No arguments provided..")
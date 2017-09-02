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
from data.vxCVtypes.insert_cv_ref_to_mysql import insert_cv_ref_to_mysql



# # Data Prep phase 7
# # For splitting into train/test
# # python3 DATAPREPPhase7.py '/u01/bigdata/01b_d2v/032/edu/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/edu/trainset.csv' '/u01/bigdata/01b_d2v/032/edu/testset.csv' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu'
# # python3 DATAPREPPhase7.py '/u01/bigdata/01b_d2v/032/skills/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/skills/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/skills/trainset.csv' '/u01/bigdata/01b_d2v/032/skills/testset.csv' '/u01/bigdata/01b_d2v/032/skills/doc2vecSkills'
# # python3 DATAPREPPhase7.py '/u01/bigdata/01b_d2v/032/personaldetails/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/personaldetails/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/personaldetails/trainset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/testset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails'
# # python3 DATAPREPPhase7.py '/u01/bigdata/01b_d2v/032/workexp/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/workexp/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/workexp/trainset.csv' '/u01/bigdata/01b_d2v/032/workexp/testset.csv' '/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp'
# print(len(sys.argv))
if(len(sys.argv)==6):
    logfile = sys.argv[1]
    srcdataset = sys.argv[2]
    desttrainset = sys.argv[3]
    desttestset = sys.argv[4]
    appd2vSrcFolder = sys.argv[5]

    print('logfile: ',logfile)
    print('srcdataset: ', srcdataset)
    print('desttrainset: ', desttrainset)
    print('desttestset: ', desttestset)
    print('appd2vSrcFolder: ', appd2vSrcFolder)

    if(logfile=='' or srcdataset=='' or desttrainset=='' or desttestset==''):
        print("No arguments provided..")
    else:


        phase7=insert_cv_ref_to_mysql(logFile=logfile)
        phase7.saveTrainTestSet(datasetFilename=srcdataset, trainsetFilename=desttrainset, testsetFilename=desttestset, appd2vSrcFolder=appd2vSrcFolder)
        # cv_cat_infer=CV_CategoryInference(logfile=logfile)
        # cv_cat_infer.saveTrainTestSet(datasetFilename=srcdataset, trainsetFilename=desttrainset, testsetFilename=desttestset, appd2vSrcFolder=appd2vSrcFolder)
else:
    print("python3 DATAPREPPhase7.py '/u01/bigdata/01b_d2v/032/edu/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/edu/trainset.csv' '/u01/bigdata/01b_d2v/032/edu/testset.csv' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu'")

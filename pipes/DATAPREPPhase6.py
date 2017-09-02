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

# Data Prep phase 6
# For joining
# python3 DATAPREPPhase6.py '/u01/bigdata/01b_d2v/032/edu/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/edu/summary_32_edu.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv'
# python3 DATAPREPPhase6.py '/u01/bigdata/01b_d2v/032/skills/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/skills/summary_32_skills.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/skills/joind2vSummaryCVpath.csv'
# python3 DATAPREPPhase6.py '/u01/bigdata/01b_d2v/032/personaldetails/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/personaldetails/summary_32_personaldetails.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/personaldetails/joind2vSummaryCVpath.csv'
# python3 DATAPREPPhase6.py '/u01/bigdata/01b_d2v/032/workexp/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/workexp/summary_32_workexp.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/workexp/joind2vSummaryCVpath.csv'
if(len(sys.argv)==5):
    logfile = sys.argv[1]
    appD2vSummaryFilename = sys.argv[2]
    cv_ref_filename = sys.argv[3]
    destCSV = sys.argv[4]

    print('logfile: ',logfile)
    print('appD2vSummaryFilename: ', appD2vSummaryFilename)
    print('cv_ref_filename: ', cv_ref_filename)
    print('destCSV: ', destCSV)
    if(logfile=='' or appD2vSummaryFilename=='' or cv_ref_filename=='' or destCSV==''):
        print("No arguments provided..")
    else:
        phase6=insert_cv_ref_to_mysql(logFile=logfile)
        # cv_cat_infer=CV_CategoryInference(logfile=logfile)
        phase6.joinCVREF_APPD2V(appD2vSummaryFilename=appD2vSummaryFilename,cv_ref_filename=cv_ref_filename,destCSV=destCSV)
        # cv_cat_infer.joinCVREF_APPD2V(appD2vSummaryFilename=
        #                               appD2vSummaryFilename,cv_ref_filename=cv_ref_filename,destCSV=destCSV)
else:
    print("Usage: python3 DATAPREPPhase6.py 'logfilename' 'appD2vSummaryFilename' 'cv_ref_filename' 'destCSV'")

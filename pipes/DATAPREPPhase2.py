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
from data.vxCVtypes import insert_cv_ref_to_mysql

#python3 DATAPREPPhase2.py '/u01/bigdata/00_appcvref/cv_ref_32.csv' '/u01/bigdata/00_appcvref/log_refine_cvref_32.log'
#python3 DATAPREPPhase2.py '/u01/bigdata/00_appcvref/cv_ref_48.csv' '/u01/bigdata/00_appcvref/log_refine_cvref_48.log'
#python3 DATAPREPPhase2.py '/u01/bigdata/00_appcvref/cv_ref_51.csv' '/u01/bigdata/00_appcvref/log_refine_cvref_51.log'
#python3 DATAPREPPhase2.py '/u01/bigdata/00_appcvref/cv_ref_200.csv' '/u01/bigdata/00_appcvref/log_refine_cvref_200.log'

if(len(sys.argv)==3):
    hostname = 'localhost'
    username = 'kah1'
    password = 'kahsiong1979'
    database = 'bigdata'
    port = 3306

    csvfilename = sys.argv[1]
    loglocfile = sys.argv[2]

    print('csvfilename: ',csvfilename)
    print('loglocfile: ', loglocfile)

    if(loglocfile=='' or loglocfile=='' ):
        print("No arguments provided..")
    else:
        dataphase2=insert_cv_ref_to_mysql.insert_cv_ref_to_mysql(hostname=hostname,username=username,password=password,database=database,port=port)
        dataphase2.refine_cvref(csvfilename=csvfilename,loglocfile=loglocfile)
        dataphase2.closeConn()
        # appform=AppToSections('/u01/bigdata/d2v','/u01/bigdata/appjson/032/')
else:
    print("No arguments provided..")
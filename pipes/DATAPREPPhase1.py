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

#python3 DATAPREPPhase1.py '032' '/u01/bigdata/00_appcvref/cv_ref_32.csv' '/u01/bigdata/00_appcvref/log_updatecvref_032.log'
#python3 DATAPREPPhase1.py '048' '/u01/bigdata/00_appcvref/cv_ref_48.csv' '/u01/bigdata/00_appcvref/log_updatecvref_048.log'
#python3 DATAPREPPhase1.py '051' '/u01/bigdata/00_appcvref/cv_ref_51.csv' '/u01/bigdata/00_appcvref/log_updatecvref_051.log'
#python3 DATAPREPPhase1.py '200' '/u01/bigdata/00_appcvref/cv_ref_200.csv' '/u01/bigdata/00_appcvref/log_updatecvref_200.log'
# Change the content here as required.
if(len(sys.argv)==4):
    hostname = 'localhost'
    username = 'kah1'
    password = 'kahsiong1979'
    database = 'bigdata'
    port = 3306

    clientid = sys.argv[1]
    scvLoc = sys.argv[2]    #Save locationo f results
    logLoc = sys.argv[3]

    print('clientid: ',clientid)
    print('scvLoc: ', scvLoc)
    print('logLoc: ', logLoc)
    if(clientid=='' or scvLoc=='' or logLoc==''):
        print("No arguments provided..")
    else:
        dataphase1=insert_cv_ref_to_mysql.insert_cv_ref_to_mysql(hostname=hostname,username=username,password=password,database=database,port=port)

        dataphase1.updateCVREF([int(clientid)], scvLoc,logLoc, directoryToProbe='/home/kah1/vX/CV')
        dataphase1.closeConn()
        # appform=AppToSections('/u01/bigdata/d2v','/u01/bigdata/appjson/032/')
else:
    print("No arguments provided..")
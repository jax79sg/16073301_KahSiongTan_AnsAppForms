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
from data.extractFromAppJSON.appToSections import AppToSections

#python3 DATAPREPPhrase3.py '/u01/bigdata/01a_d2v_QandA' '/u01/bigdata/00_appjson/200/' '/u01/bigdata/01a_d2v_QandA/log_appToSection32.log'
if(len(sys.argv)==4):
    d2vLoc = sys.argv[1]
    srcLoc = sys.argv[2]
    logLoc = sys.argv[3]

    print('d2vloc: ',d2vLoc)
    print('srcloc: ', srcLoc)
    print('logloc: ', logLoc)
    if(d2vLoc=='' or srcLoc=='' or logLoc==''):
        print("No arguments provided..")
    else:
        appform = AppToSections(d2vLoc, srcLoc, logLoc)
else:
    print("No arguments provided..")
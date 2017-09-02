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
from data.raters.CVSectionRater import CVSectionRater

#python3 /home/kah1/remote_cookie_runtime/src/pipes/raters.py '/u01/bigdata/03b_raters/log_raters.log' '/u01/bigdata/03b_raters/Agreementkahsiong_01.csv,/u01/bigdata/03b_raters/Agreementcharlotte_02.csv,/u01/bigdata/03b_raters/Agreementxiong_03.csv,/u01/bigdata/03b_raters/Agreementkahsiongtwo_04.csv,/u01/bigdata/03b_raters/Agreementmin_05.csv'
if(len(sys.argv)==3):
    logLoc = sys.argv[1]
    ratersFiles = sys.argv[2]

    print('logloc: ', logLoc)
    print('ratersFiles: ',ratersFiles)
    rater = CVSectionRater(logFilename=logLoc)
    rater.buildRaterRecords(ratersFiles)
    print(rater.computeFlessisKappa())

else:
    print("No arguments provided..")
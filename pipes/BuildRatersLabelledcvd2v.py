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

#python3 /home/kah1/remote_cookie_runtime/src/pipes/BuildRatersLabelledcvd2v.py '/u01/bigdata/03b_raters/client291/log_buildRatersLabelledcvd2v.log' '/u01/bigdata/03b_raters/client291/ratersrecords' '/u01/bigdata/03b_raters/client291/majorityVotedRatersCvd2v'
if(len(sys.argv)==4):
    logLoc = sys.argv[1]
    folderRatersOwnRecords = sys.argv[2]
    dstFolder = sys.argv[3]

    print('logloc: ', logLoc)
    print('folderRatersOwnRecords: ',folderRatersOwnRecords)
    print('dstFolder: ', dstFolder)

    rater = CVSectionRater(logFilename=logLoc)
    rater.generateRaterLabelledCVd2v(folderRatersOwnRecords=folderRatersOwnRecords, dstFolder=dstFolder)

else:
    print("No arguments provided..")
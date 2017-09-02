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
from data.extractFromCV.CVLabels import CVLabels

# # TestA phase 1xa
# #python3 CVLabels.py '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/log_labeltrain.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '40' '5' '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/train/CV_032_Extracts.csv'  '/home/kah1/vX/CV' '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/train/raw' /u01/bigdata/03a_01b_test/cvd2v/032/labelled/train/cvd2v'
# #python3 CVLabels.py '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/log_labeltest.log' '/u01/bigdata/03a_01b_test/cvd2v/032/test' '40' '5' '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/test/CV_032_Extracts.csv'  '/home/kah1/vX/CV' '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/test/raw' '/u01/bigdata/03a_01b_test/cvd2v/032/labelled/test/cvd2v'
#python3 PREPARE_CVSectionForLabelling.py '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/log_labeltest.log' '/u01/bigdata/03a_01b_test/cvd2v/200/test' '40' '5' '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/test/CV_200_Extracts.csv'  '/home/kah1/vX/CV' '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/test/raw' '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/test/cvd2v'
#python3 PREPARE_CVSectionForLabelling.py '/u01/bigdata/03a_01b_test/cvd2v/291/labelled/log_291labeltest.log' '/u01/bigdata/03a_01b_test/cvd2v/291/test' '40' '5' '/u01/bigdata/03a_01b_test/cvd2v/291/labelled/test/CV_291_Extracts.csv'  '/home/kah1/vX/CV' '/u01/bigdata/03a_01b_test/cvd2v/291/labelled/test/raw' '/u01/bigdata/03a_01b_test/cvd2v/291/labelled/test/cvd2v'
if __name__ == "__main__":
    if(len(sys.argv)==9):
        logFile = sys.argv[1]
        folderContainingCVD2V = sys.argv[2]
        maxNoOfCVs=int(sys.argv[3])    # It will only process the first maxNoOfCVs
        breakSize=int(sys.argv[4])
        saveFilename=sys.argv[5]
        rawFolder=sys.argv[6]
        dstRawFolder=sys.argv[7]
        dstCvd2vFolder = sys.argv[8]
        print('Logging to ', logFile)
        print('folderContainingCVD2V',folderContainingCVD2V)
        print('maxNoOfCVs', maxNoOfCVs)
        print('breakSize', breakSize)
        print('saveFilename', saveFilename)
        print('rawFolder', rawFolder)
        print('dstRawFolder', dstRawFolder)
        print('dstCvd2vFolder', dstCvd2vFolder)


        cvlbl=CVLabels(logFile)
        cvlbl.createLabels(folderContainingCVD2V=folderContainingCVD2V, maxNoOfCVs=maxNoOfCVs, breakSize=breakSize,
                           saveFilename=saveFilename, rawFolder=rawFolder, dstRawFolder=dstRawFolder,dstCvd2vFolder=dstCvd2vFolder)
        # cvlbl.transferOriginals(rawFolder=rawFolder,dstFolder=dstRawFolder,dataframe=df)
    else:
        print('No arguments given')
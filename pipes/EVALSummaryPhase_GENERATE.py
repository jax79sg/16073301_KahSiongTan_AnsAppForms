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
from models import Evaluator
# Change the content here as required.


if __name__ == "__main__":
    if(len(sys.argv)==3):
        logFile = sys.argv[1]
        folderOfMetricFiles=sys.argv[2]

        print('logFile:',logFile)
        print('folderOfMetricFiles:', folderOfMetricFiles)


        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        eval = Evaluator.Evaluator(utilObj=util)
        eval.generateSummary(folderOfMetricFiles)
    else:
        print("No arguments provided..")
        exit(-1)
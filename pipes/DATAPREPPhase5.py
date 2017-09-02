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
from data.extractFromAppJSON.SummaryAppSections import SummaryAppSections

## python3 DATAPREPPhase5.py '/u01/bigdata/01b_d2v/032/edu/summary_32_edu.log' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu' 'edu' '/u01/bigdata/01b_d2v/032/edu/summary_32_edu.csv'
## python3 DATAPREPPhase5.py '/u01/bigdata/01b_d2v/032/workexp/summary_32_workexp.log' '/u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp' 'workexp' '/u01/bigdata/01b_d2v/032/workexp/summary_32_workexp.csv'
if(len(sys.argv)==5):
    logfile = sys.argv[1]
    srcFolder = sys.argv[2]
    categoryName = sys.argv[3]
    resultsFilename = sys.argv[4]

    print('logfile: ',logfile)
    print('srcFolder: ', srcFolder)
    print('categoryName: ', categoryName)
    print('resultsFilename: ', resultsFilename)
    if(logfile=='' or srcFolder=='' or categoryName=='' or resultsFilename==''):
        print("No arguments provided..")
    else:
        summary=SummaryAppSections(logfile=logfile)
        summary.createSummary(categoryName=categoryName,resultsFilename=resultsFilename,srcFolder=srcFolder)
else:
    print("Usage: python3 SummaryAppSections 'logfilename' 'srcFolder' 'categoryName' 'resultsFilename'")
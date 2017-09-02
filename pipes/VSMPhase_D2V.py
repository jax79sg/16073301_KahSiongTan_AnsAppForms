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
from commons import Utilities
from features.D2V import D2V

# Change the content here as required.
# python3 /home/kah1/remote_cookie_runtime/src/pipes/featurePhase4_D2V.py '/u01/bigdata/02d_d2vModel1/log_cvD2v100VectorSpaceModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '/u01/bigdata/02d_d2vModel1/cvD2v100VectorSpaceModel.model' '1' '10000000' '100'
# python3 /home/kah1/remote_cookie_runtime/src/pipes/featurePhase4_D2V.py '/u01/bigdata/02d_d2vModel1/log_cvD2v300VectorSpaceModel.log' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '/u01/bigdata/02d_d2vModel1/cvD2v300VectorSpaceModel.model' '1' '10000000' '300'
if __name__ == "__main__":
    if(len(sys.argv)==8):
        logFile = sys.argv[1]
        folderListOfCorpus=sys.argv[2]
        dstFilename=sys.argv[3]
        ngram=int(sys.argv[4])
        maxdocs = int(sys.argv[5])
        maxDim = int(sys.argv[6])
        tokenRules = (sys.argv[7])


        print('logFile:',logFile)
        print('folderListOfCorpus:', folderListOfCorpus)
        print('dstFilename:', dstFilename)
        print('ngram:', ngram)
        print('maxdocs:', maxdocs)
        print('maxDim:', maxDim)
        print('tokenRules:', tokenRules)

        util=Utilities.Utility()
        util.setupLogFileLoc(logFile=logFile)
        util.setupTokenizationRules(tokenRules)

        w2v = D2V(utilObj=util,wordFreqIgnored=2, vectordim=maxDim)
        w2v.buildCorpus(folderListOfCorpus=folderListOfCorpus, ngram=ngram, maxdocs=maxdocs,
                        dstFilename=dstFilename, maxDim=maxDim)
        vetor = w2v.inferVector('Non existing sebastian developing morning')
        vetor2 = w2v.inferVector('morning')
        vetor3 = w2v.inferVector('jax')
        util.logInfo('D2V', 'Vectorsize:' + str(len(vetor)))
        print(vetor2)
        print(vetor3)
        from sklearn.metrics.pairwise import cosine_similarity

        print(cosine_similarity(vetor, vetor2))
        print(cosine_similarity(vetor, vetor3))
        print(cosine_similarity(vetor2, vetor3))

        d2v2 = D2V(utilObj=util)
        d2v2.loadVectorSpaceModel(dstFilename)
        vetor = d2v2.inferVector('Non existing sebastian developing morning')
        vetor2 = d2v2.inferVector('morning')
        vetor3 = d2v2.inferVector('jax')
        print(vetor)
        print(vetor2)
        print(vetor3)
        from sklearn.metrics.pairwise import cosine_similarity

        print(cosine_similarity(vetor, vetor2))
        print(cosine_similarity(vetor, vetor3))
        print(cosine_similarity(vetor2, vetor3))

    else:
        print("No arguments provided..")
        exit(-1)
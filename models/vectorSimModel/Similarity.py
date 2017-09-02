"""
Compute pairwise similarity
"""

from sklearn.metrics.pairwise import cosine_similarity
from commons import Utilities

class Similarity():
    _util=None

    def __init__(self, utilObj=None, logFile=None):
        if (utilObj != None):
            self._util = utilObj
        elif (logFile != None):
            self._util = Utilities.Utility()
            self._util.setupLogFileLoc(logFile)
        pass

    def cosine_sim(self, Xtrain, Xtest):
        """
        Compute cosine similarity between each element of Xtrain and Xtest
        :param Xtrain: Vector size N
        :param Xtest: Vector size N
        :return: Array of similarity results
        """
        # print('AppData Xtrain:',Xtrain)
        # print('CvData Xtest:', Xtest)
        results = cosine_similarity(X=Xtrain,Y=Xtest)
        self.validateSimMatrixShape(results,Xtrain,Xtest)
        # print('Sim matrix:',results)
        return results

    def validateSimMatrixShape(self, results, Xtrain, Xtest):
        validateResult=True
        xShape=len(Xtrain)
        yShape=len(Xtest)
        if (results.shape[0]!=xShape or results.shape[1]!=yShape):
            self._util.logError('Similarity','Expecting [',+str(xShape)+','+str(yShape)+'] but get ' + results.shape + ' instead')
            validateResult=False
        return validateResult






from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from commons import Utilities

class ClassiferLoader():

    TYPE_SVM='svm'
    TYPE_MULTINORM_NAIVEBAYES='mbn'

    classifier=None
    X=None
    Y=None
    util=None
    fitted=False

    def __init__(self, type=None, utilObj=None, utilName=None):
        if (utilObj!=None):
            self.util=utilObj
        elif(utilName!=None):
            selfutil = Utilities.Utility()
            self.util.setupLogFileLoc(utilName)


        if (type==self.TYPE_SVM):
            self.classifier=SVC()
        elif(type==self.TYPE_MULTINORM_NAIVEBAYES):
            self.classifier=MultinomialNB()
        pass

    def loadClassiferParams(self, **arg):
        pass

    def loadFeaturesLabels(self,X,Y):
        self.X=X
        self.Y=Y
        pass

    def trainClassifier(self):
        if (self.classifier==None or self.X==None or self.Y==None):
            self.util.logDebug('ClassifierLoader', 'Classifier not initialised properly. Check the classiiferload and feature load')
        else:
            self.classifier.fit(self.X,self.Y)
            self.fitted=True
            self.util.logDebug('ClassifierLoader',
                               'Data fitted')

    def predict(self,x):
        result=None
        if self.fitted==True:
            results=self.classifier.predict(x)
        else:
            self.util.logDebug('ClassiferLoader', 'Classifier has yet to be trained, prediction not possible')
        return results


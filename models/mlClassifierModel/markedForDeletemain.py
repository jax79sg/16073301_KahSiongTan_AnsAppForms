
"""
- Single training appd2v doc2vec model
    - Generate a doc2vec model
        - Dump all training app2v (regardless of categories) to generate model
    -
"""

from models.mlClassifierModel import FeatureLabelLoader
from models.mlClassifierModel import markedForDeleteClassifierLoader
from commons import Utilities
import numpy as np
if __name__ == "__main__":
    LABEL_EDU=0
    LABEL_SKILLS=1
    LABEL_PERSONAL=2
    LABEL_WORKEXP=3
    util = Utilities.Utility()
    util.setupLogFileLoc('/home/kah1/logme.log')
    featLoader=FeatureLabelLoader.FeatureLabelLoader(util)
    npXskills,npYskills=featLoader.loadTrainDoc2vecFeatureLabels('/home/kah1/train32Skills.model', LABEL_SKILLS)
    npXedu, npYedu = featLoader.loadTrainDoc2vecFeatureLabels('/home/kah1/train32edu.model', LABEL_EDU)
    npXpersonal, npYpersonal = featLoader.loadTrainDoc2vecFeatureLabels('/home/kah1/train32personal.model', LABEL_PERSONAL)
    npXworkexp, npYworkexp = featLoader.loadTrainDoc2vecFeatureLabels('/home/kah1/train32workexp.model', LABEL_WORKEXP)

    npXtrain=np.append(npXskills, axis=1)
    npYtrain=np.append(npYskills, axis=1)

    npXtrain=np.append(npXedu, axis=1)
    npYtrain=np.append(npYedu, axis=1)

    npXtrain=np.append(npXpersonal, axis=1)
    npYtrain=np.append(npYpersonal, axis=1)

    npXtrain=np.append(npXworkexp, axis=1)
    npYtrain=np.append(npYworkexp, axis=1)

    npXtrain, npYtrain=featLoader.unifiedShuffle(npXtrain, npYtrain)

    classifier=markedForDeleteClassifierLoader.ClassiferLoader(util)
    classifier.loadFeaturesLabels(npXtrain, npYtrain)
    classifier.trainClassifier()


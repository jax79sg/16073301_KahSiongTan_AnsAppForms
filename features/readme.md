The role of features here is to 
1. Given the label and data of appd2v, generate a feature vector and save the data.

The features here should follow the following interface rules.
This would be called by one of the classes in the models package.
Interface
- Methods
    - init(logfilename, utilObj)
        - Initialise utility class and whatever
    - buildCorpus(folderListOfCorpus=None,ngram, maxdocs,dstFilename, maxDim)
        - Build/Fit the training vector space model based on all the documents in folderListOfCorpus.
        - Limit to a training document size of maxdocs
        - Save the fitted feature model in dstFilename
    - inferVector (strList)
        - For online inference of new document string (raw string).
    - loadVectorSpaceModel(srcFilename)
        - Load a previously saved feature model
    - saveVectorSpaceModel(dstFilename)
        - Save the fitted feature model into a file
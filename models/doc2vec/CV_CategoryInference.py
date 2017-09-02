
import sys
sys.path.append("/home/kah1/remote_cookie_runtime")
sys.path.append("/home/kah1/remote_cookie_runtime/src")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromAppJSON")
sys.path.append("/home/kah1/remote_cookie_runtime/src/data/extractFromCV")
sys.path.append("/home/kah1/remote_cookie_runtime/src/models/doc2vec")
import glob
import pandas as pd
import os
from commons import Utilities
from models.doc2vec.doc2vecFrame import doc2vecFrame

class CV_CategoryInference():
    _util=None
    _logfile=None

    def __init__(self,logfile=None):
        self._logfile=logfile
        self._util = Utilities.Utility()
        self._util.setupLogFileLoc(logfile)


    def inferMatches(self, eduModelFilename=None,skillModelFilename=None, personalDetailModelFilename=None, cvD2vFolder=None, resultsFilename=None):
        """
        - For each CVd2v,
        -- Perform inference and category matching
        1a. Infer a vector from the EDU model/category
        1b. Retrieve top 3 most similar documents from EDU model (Normalise to 100%)
        1c. Average the similarity scores from the top 3.
        Repeat above for skills and personaldetails

        4a. Discard categories with scores less than 40% (Observations show that random scores is around 35 to 40)
        4b. Tag remaining categories and their average scores to the CV portion.
        4c. Save the results as csv
            --> categories (seperated by semi-colons), appid, cv_extract_filename
        -- Save results (clientid, appid, cvd2vFullpath, categories, scores)
        :param eduModelFilename:
        :param skillModelFilename:
        :param personalDetailModelFilename:
        :param cvD2vFolder:
        :param resultsFilename:
        :return:
        """
        doc2vecframeEdu = doc2vecFrame(logFilename=self._logfile)
        doc2vecframeEdu.loadModel(eduModelFilename)
        doc2vecframeSkills = doc2vecFrame(logFilename=self._logfile)
        doc2vecframeSkills.loadModel(skillModelFilename)
        doc2vecframePersonalDetails = doc2vecFrame(logFilename=self._logfile)
        doc2vecframePersonalDetails.loadModel(personalDetailModelFilename)

        counter=0
        errcounter=0
        self._util.logDebug('CV_CategoryInference-inferMatches', 'Gathering file information from ' + cvD2vFolder + '...this will take a while')
        noOfFiles = sum([len(files) for r, d, files in os.walk(cvD2vFolder)])
        self._util.logDebug('CV_CategoryInference-inferMatches',
                             'No of files found: ' + str(noOfFiles) + '\nProcessing now')

        fullResults = pd.DataFrame(columns=('appid', 'clientid', 'cvd2vfilename', 'categories', 'scores', 'content'))
        for cvD2VFile in sorted(glob.iglob(cvD2vFolder + '**/*.cvd2v', recursive=True)):  # Recursively probe for cvd2v files

        # try:
            fileContent = open(cvD2VFile, 'r').read()
            fileContentTokens=self._util.tokenize(fileContent)

            scoreEdu=doc2vecframeEdu.getAvgSimilarity(fileContentTokens)
            scoreSkills = doc2vecframeSkills.getAvgSimilarity(fileContentTokens)
            scorePersonalDetails = doc2vecframePersonalDetails.getAvgSimilarity(fileContentTokens)

            #Pack similarity scores in a dataframe for manipulation
            edurow = pd.DataFrame(data={'score': [scoreEdu], 'category': ['edu']})
            skillsrow = pd.DataFrame(data={'score': [scoreSkills], 'category': ['skills']})
            personaldetailsrow = pd.DataFrame(data={'score': [scorePersonalDetails], 'category': ['personaldetails']})
            simScores = pd.DataFrame(columns=('score', 'category'))
            simScores = simScores.append(edurow)
            simScores = simScores.append(skillsrow)
            simScores = simScores.append(personaldetailsrow)
            simScores.reset_index(inplace=True)
            # print('Before trim:',simScores)
            # print(simScores[simScores.score < 0.45].index)
            simScores = simScores.drop(simScores[simScores.score < 0.45].index)  #Drop scores less than 0.45.
            simScores=simScores.sort_values(by='score',axis=0,ascending=False)  #Sort with higher scores first
            # print('After trim:',simScores)
            simScores.reset_index(inplace=True)
            #Pack results(Get clientid, appid from cvd2v filename)
            appid=cvD2VFile.split('_')[-2]
            clientid = cvD2VFile.split('_')[-3]
            cvd2vfilename=cvD2VFile
            categories=''
            scores=''
            counterItem=0
            for index, item in simScores.iterrows():
                if counterItem==0:
                    categories=item['category']
                    scores = str(item['score'])
                else:
                    categories = categories+':'+item['category']
                    scores=scores+':'+str(item['score'])
                counterItem=counterItem+1
            newEntry = pd.DataFrame(data={'appid': [appid], 'clientid': [clientid],'cvd2vfilename': [cvd2vfilename], 'categories': [categories], 'scores': [scores], 'content':[fileContent]})
            fullResults = fullResults.append(newEntry)


            counter=counter+1
            if counter % 1000 == 0:
                self._util.logDebug('CV_CategoryInference-inferMatches',
                                    str(counter) + ' out of ' + str(noOfFiles) + ' files completed with ' + str(
                                         errcounter) + ' errors.')
                self._util.logDebug('CV_CategoryInference-inferMatches', 'Saving ' + str(fullResults.shape[0]) + ' inferred results!')
                saveAsFilename=(resultsFilename.split('.')[0]+'_'+str(counter).zfill(1)+'.'+resultsFilename.split('.')[-1])
                fullResults.to_csv(saveAsFilename, ',', mode='a', header=False, index=False,
                                   columns=['appid', 'clientid', 'cvd2vfilename','content', 'categories', 'scores'])
                fullResults = None
                fullResults = pd.DataFrame(columns=('appid', 'clientid', 'cvd2vfilename', 'categories', 'scores', 'content'))
        # except Exception as error:
        #     errcounter =errcounter+1
        #     self._util.logDebug('CV_CategoryInference-inferMatches','Error encountered: '+ repr(error))
        self._util.logDebug('CV_CategoryInference-inferMatches', 'Final save!')
        counter=counter+1
        saveAsFilename = (
        resultsFilename.split('.')[0] + '_' + str(counter).zfill(1) + '.' + resultsFilename.split('.')[-1])
        fullResults.to_csv(saveAsFilename, ',', mode='a', header=False, index=False,
                           columns=['appid', 'clientid', 'cvd2vfilename','content', 'categories', 'scores'])


    # def saveTrainTestSet(self, datasetFilename=None, trainsetFilename=None, testsetFilename=None, appd2vSrcFolder=None):
    #     """
    #     Split and save into training and test sets
    #     :param datasetFilename: E.g. joind2vSummaryCVpath.csv
    #     :param trainsetFilename:
    #     :param testsetFilename:
    #     :param Appd2vSrcFolder: E.g. /u01/bigdata/01b_d2v/032/edu/doc2vecEdu/
    #     :param clientid:
    #     :return:
    #     """
    #     dataset = pd.read_csv(datasetFilename)
    #     # print(dataset)
    #
    #     self._util.logDebug('CV_CategoryInference-saveTrainTestSet',
    #                         'Splittng ' + datasetFilename + '...')
    #     trainset, testset = self.__splitTrainTestset(dataset=dataset,trainPercent=80)
    #     self._util.logDebug('CV_CategoryInference-saveTrainTestSet',
    #                         'Split completed...')
    #
    #     self._util.logDebug('CV_CategoryInference-saveTrainTestSet',
    #                         'Saving...')
    #     trainset.to_csv(trainsetFilename, ',', mode='w', header=True, index=False, columns=['appid', 'clientid', 'fullfilename'])
    #     testset.to_csv(testsetFilename, ',', mode='w', header=True, index=False, columns=['appid', 'clientid', 'fullfilename'])
    #     self._util.logDebug('CV_CategoryInference-saveTrainTestSet',
    #                         'Saving completed...')
    #
    #     self.__copyDataToFolder(dataset=trainset, appd2vSrcFolder=appd2vSrcFolder, destFolder='train')
    #     self.__copyDataToFolder(dataset=testset, appd2vSrcFolder=appd2vSrcFolder, destFolder='test')
    #     self._util.logDebug('CV_CategoryInference-saveTrainTestSet',
    #                         'Operation completed...')
    #
    # def __copyDataToFolder(self, dataset=None, appd2vSrcFolder=None, destFolder=None):
    #     counter=0
    #     if os.path.exists(appd2vSrcFolder + '/' + destFolder):
    #         print('This is a destructive action, rmtree ' + appd2vSrcFolder + '/' + destFolder + '?\n please confirm with Y')
    #         if (input('> ')=='Y'):
    #             if os.path.isdir(appd2vSrcFolder + '/' + destFolder):
    #                 shutil.rmtree(appd2vSrcFolder + '/' + destFolder)
    #             else:
    #                 os.remove(appd2vSrcFolder + '/' + destFolder)
    #         else:
    #             self._util.logDebug('CV_CategoryInference__copyDataToFolder', 'User aborted')
    #             return (-1)
    #     os.makedirs(appd2vSrcFolder + '/' + destFolder)
    #
    #     for index, row in dataset.iterrows():
    #         appid = row['appid']
    #         clientid = row['clientid']
    #         fullfilename = row['fullfilename'] #This filename has pdf extensions
    #
    #         #Copy all the category's clientid's appid's d2v into the train subfolder of trainsetFilename
    #         srcFile=appd2vSrcFolder+'/app_'+str(clientid).zfill(3)+'_'+str(appid)+'.d2v'
    #         # self._util.logDebug('CV_CategoryInference-moveTrainDataToTrainFolder',
    #         #                     'Copying ' + srcFile + ' to ' + appd2vSrcFolder + '/' + destFolder)
    #
    #
    #
    #         copiedPath=shutil.copy2(srcFile,appd2vSrcFolder+'/'+destFolder)
    #         # os.system("cp %s %s" % (srcFile,appd2vSrcFolder+'/'+destFolder))
    #         # self._util.logDebug('CV_CategoryInference-moveTrainDataToTrainFolder','Copied as '+copiedPath)
    #         counter=counter+1
    #         if(counter%1000==0):
    #             self._util.logDebug('CV_CategoryInference-moveTrainDataToTrainFolder',
    #                                 'Copied ' + str(counter) + ' files to folder:' + appd2vSrcFolder + '/' + destFolder)
    #
    # def __splitTrainTestset(self,dataset=None,trainPercent=80, maxTrainSize=50000):
    #     """
    #
    #     :param dataset: Must be a Dataframe
    #     :param trainPercent:
    #     :return:
    #     """
    #     dataset=dataset.sample(frac=1) #Shuffled
    #     sizeofDataset=len(dataset)
    #     sizeOfTrain=int((trainPercent/100)*sizeofDataset)
    #     self._util.logDebug('CV_CategoryInference-splitTrainTestset',
    #                         'Computed train size: ' + str((sizeOfTrain)))
    #     if (sizeOfTrain>maxTrainSize):
    #         sizeOfTrain=maxTrainSize
    #     trainDataset=dataset.iloc[:sizeOfTrain]
    #     testDataset=dataset.iloc[sizeOfTrain:-1]
    #     self._util.logDebug('CV_CategoryInference-splitTrainTestset', 'Full size of ' + str(sizeofDataset) + ' splitted into train:' + str(len(trainDataset)) + ' and test:' + str(len(testDataset)))
    #     return trainDataset,testDataset



    # def joinCVREF_APPD2V(self, appD2vSummaryFilename=None, cv_ref_filename=None, destCSV=None):
    #     """
    #
    #     Perform leftjoin of df_appD2vSummary to df_cvref
    #     Remove duplicates
    #     Remove those that cannot relate to a CV
    #     df_appD2vSummary
    #     (columns=('category', 'clientid', 'appid'))
    #
    #     df_cvref
    #     (columns=['cv_ref_id', 'appid', 'candidateid', 'oppid', 'cfilename', 'clientid', 'isfileexists', 'iscontentenglish',
    #     'fullfilename'])
    #
    #
    #     :param appD2vSummaryFilename:
    #     :param cv_ref_filename:
    #     :param destCSV:
    #     :return:
    #     """
    #     self._util.startTimeTrack()
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V', 'Loading ' + appD2vSummaryFilename)
    #     df_appD2vSummary = pd.read_csv(appD2vSummaryFilename, header=None)
    #     df_appD2vSummary.columns = ['category', 'clientid', 'appid']
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V', 'Loaded ' + appD2vSummaryFilename + ' in ' + self._util.checkpointTimeTrack())
    #
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V', 'Loading ' + cv_ref_filename)
    #     df_cvref = pd.read_csv(cv_ref_filename, header=None)
    #     df_cvref.columns = ['cv_ref_id', 'appid', 'candidateid', 'oppid', 'cfilename', 'clientid', 'isfileexists',
    #                'iscontentenglish',
    #                'fullfilename']
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Loaded ' + cv_ref_filename + ' in ' + self._util.checkpointTimeTrack())
    #
    #
    #     #Drop unnecessary columns
    #     df_cvref = df_cvref.drop('cv_ref_id', 1)
    #     df_cvref = df_cvref.drop('candidateid', 1)
    #     df_cvref = df_cvref.drop('oppid', 1)
    #     df_cvref = df_cvref.drop('cfilename', 1)
    #     df_cvref = df_cvref.drop('isfileexists', 1)
    #     df_cvref = df_cvref.drop('iscontentenglish', 1)
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V', 'Size of left: ' + str(df_appD2vSummary.shape[0]) + ' Size of right: ' + str(df_cvref.shape[0]) + ' Merging.... ')
    #     mergedDF=pd.merge(df_appD2vSummary,df_cvref, how='left', on=['clientid','appid'], )
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Merged completed..Size of merged: ' + str(mergedDF.shape[0]) + ' in ' + self._util.checkpointTimeTrack())
    #
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Dropping rows without cv filepaths...')
    #     mergedDF = mergedDF.replace(np.nan, '-1', regex=True)
    #     mergedDF=mergedDF[mergedDF.fullfilename!='-1']
    #
    #     mergedDF=mergedDF.sort_values(by='appid')
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Drop completed..Size of post drop: ' + str(mergedDF.shape[0]) + ' in ' + self._util.checkpointTimeTrack())
    #
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Dropping duplicated rows ...')
    #     mergedDF=mergedDF.drop_duplicates()
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Dedup completed..Size of post dedup: ' + str(
    #                             mergedDF.shape[0]) + ' in ' + self._util.checkpointTimeTrack())
    #
    #
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Saving to ' + destCSV)
    #     mergedDF.to_csv(destCSV, ',', mode='w', header=True, index=False, columns=['appid','clientid','fullfilename'])
    #     self._util.logDebug('CV_CategoryInference-joinCVREF_APPD2V',
    #                         'Saving completed in ' + self._util.checkpointTimeTrack())


# Data Prep phase 6
# For joining
# python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/edu/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/edu/summary_32_edu.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv'
# python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/skills/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/skills/summary_32_skills.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/skills/joind2vSummaryCVpath.csv'
# python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/personaldetails/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/personaldetails/summary_32_personaldetails.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/personaldetails/joind2vSummaryCVpath.csv'
# python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/workexp/log_joind2vSummaryCVpath.log'  '/u01/bigdata/01b_d2v/032/workexp/summary_32_workexp.csv' '/u01/bigdata/00_appcvref/cv_ref_32_filtered.csv' '/u01/bigdata/01b_d2v/032/workexp/joind2vSummaryCVpath.csv'
# if(len(sys.argv)==5):
#     logfile = sys.argv[1]
#     appD2vSummaryFilename = sys.argv[2]
#     cv_ref_filename = sys.argv[3]
#     destCSV = sys.argv[4]
#
#     print('logfile: ',logfile)
#     print('appD2vSummaryFilename: ', appD2vSummaryFilename)
#     print('cv_ref_filename: ', cv_ref_filename)
#     print('destCSV: ', destCSV)
#     if(logfile=='' or appD2vSummaryFilename=='' or cv_ref_filename=='' or destCSV==''):
#         print("No arguments provided..")
#     else:
#         cv_cat_infer=CV_CategoryInference(logfile=logfile)
#         cv_cat_infer.joinCVREF_APPD2V(appD2vSummaryFilename=appD2vSummaryFilename,cv_ref_filename=cv_ref_filename,destCSV=destCSV)
# else:
#     print("Usage: python3 CV_CategoryInference.py 'logfilename' 'appD2vSummaryFilename' 'cv_ref_filename' 'destCSV'")


# # Data Prep phase 7
# # For splitting into train/test
# # python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/edu/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/edu/trainset.csv' '/u01/bigdata/01b_d2v/032/edu/testset.csv' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu'
# # python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/skills/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/skills/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/skills/trainset.csv' '/u01/bigdata/01b_d2v/032/skills/testset.csv' '/u01/bigdata/01b_d2v/032/skills/doc2vecSkills'
# # python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/personaldetails/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/personaldetails/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/personaldetails/trainset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/testset.csv' '/u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails'
# print(len(sys.argv))
# if(len(sys.argv)==6):
#     logfile = sys.argv[1]
#     srcdataset = sys.argv[2]
#     desttrainset = sys.argv[3]
#     desttestset = sys.argv[4]
#     appd2vSrcFolder = sys.argv[5]
#
#     print('logfile: ',logfile)
#     print('srcdataset: ', srcdataset)
#     print('desttrainset: ', desttrainset)
#     print('desttestset: ', desttestset)
#     print('appd2vSrcFolder: ', appd2vSrcFolder)
#
#     if(logfile=='' or srcdataset=='' or desttrainset=='' or desttestset==''):
#         print("No arguments provided..")
#     else:
#         cv_cat_infer=CV_CategoryInference(logfile=logfile)
#         cv_cat_infer.saveTrainTestSet(datasetFilename=srcdataset, trainsetFilename=desttrainset, testsetFilename=desttestset, appd2vSrcFolder=appd2vSrcFolder)
# else:
#     print("python3 CV_CategoryInference.py '/u01/bigdata/01b_d2v/032/edu/log_splitTrainTest.log' '/u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv' '/u01/bigdata/01b_d2v/032/edu/trainset.csv' '/u01/bigdata/01b_d2v/032/edu/testset.csv' '/u01/bigdata/01b_d2v/032/edu/doc2vecEdu'")


# Test phase 2
# python3 CV_CategoryInference.py '/u01/bigdata/03a_01b_test/cvd2v/log_TestA_32_train_results.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model' '/u01/bigdata/02c_d2vModel1/train32Skills.model' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model' '/u01/bigdata/03a_01b_test/cvd2v/032/userlabelled' '/u01/bigdata/03a_01b_test/cvd2v/TestA_32_userlabelled_results.csv'
# python3 CV_CategoryInference.py '/u01/bigdata/03a_01b_test/cvd2v/log_TestA_32_test_results.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model' '/u01/bigdata/02c_d2vModel1/train32Skills.model' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model' '/u01/bigdata/03a_01b_test/cvd2v/032/test' '/u01/bigdata/03a_01b_test/cvd2v/TestA_32_test_results.csv'
# python3 CV_CategoryInference.py '/u01/bigdata/03a_01b_test/cvd2v/log_TestA_32_trial_results.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model' '/u01/bigdata/02c_d2vModel1/train32Skills.model' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model' '/u01/bigdata/03a_01b_test/cvd2v/032/trial' '/u01/bigdata/03a_01b_test/cvd2v/TestA_32_trial_results.csv'
# print(len(sys.argv))
# if(len(sys.argv)==7):
#     logfile = sys.argv[1]
#     eduModelFilename = sys.argv[2]
#     skillModelFilename = sys.argv[3]
#     personalDetailModelFilename = sys.argv[4]
#     cvD2vFolder = sys.argv[5]
#     resultsFilename = sys.argv[6]
#
#     print('logfile: ',logfile)
#     print('eduModelFilename: ', eduModelFilename)
#     print('skillModelFilename: ', skillModelFilename)
#     print('personalDetailModelFilename: ', personalDetailModelFilename)
#     print('cvD2vFolder: ', cvD2vFolder)
#     print('resultsFilename: ', resultsFilename)
#
#     if(logfile==''):
#         print("No arguments provided..")
#     else:
#         cv_cat_infer=CV_CategoryInference(logfile=logfile)
#         cv_cat_infer.inferMatches(eduModelFilename=eduModelFilename,skillModelFilename=skillModelFilename, personalDetailModelFilename=personalDetailModelFilename, cvD2vFolder=cvD2vFolder, resultsFilename=resultsFilename)
# else:
#     print("python3 CV_CategoryInference.py '/u01/bigdata/03a_01b_test/cvd2v/log_TestA_32_train_results.log' '/u01/bigdata/02c_d2vModel1/train32Edu.model' '/u01/bigdata/02c_d2vModel1/train32Skills.model' '/u01/bigdata/02c_d2vModel1/train32PersonalDetails.model' '/u01/bigdata/03a_01b_test/cvd2v/032/train' '/u01/bigdata/03a_01b_test/cvd2v/TestA_32_train_results.csv'")

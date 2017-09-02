"""
Utility class for various uses
Method name should be clear enough
"""

import datetime
import shutil
import re
import sys
import os
import langdetect
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import enchant
from langdetect import DetectorFactory
import logging
import joblib
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

class Utility():
    APPTODOC2VEC='APPTODOC2VEC'
    CVTODOC2VEC='CVTODOC2VEC'
    DOC2VECFRAME = 'DOC2VECFRAME'
    PARALOADER = 'PARALOADER'
    LOOKUP_CAT_SKILLS='skills'
    LOOKUP_CAT_EDU = 'edu'
    LOOKUP_CAT_PERSONALDETAILS = 'personaldetails'
    LOOKUP_EXT_APPD2V='d2v'
    LOOKUP_EXT_CVD2V='cvd2v'
    THRES_EDU=1
    THRES_SKILLS = 1
    THRES_PERSONALDETAILS = 3

    TYPE_CV=0
    TYPE_APP=1
    _logFile=''

    COLOR_RED='\033[91m'
    COLOR_NORMAL ='\033[0m'
    COLOR_GREEN ='\033[92m'

    removeStopwords = True  #Compulsary
    toLowercase = True #Compulsary
    replaceSlash = True #Compulsary
    flatEmail = False
    flatPhone = False
    flatYear = False
    flatMonth = False
    flatNumber = False
    lemmatize = False

    def setupTokenizationRules(self,csvRules):
        rules=csvRules.split(',')
        if ('removeStopwords' in rules):
            self.removeStopwords=True
        if ('toLowercase' in rules):
            self.toLowercase=True
        if ('replaceSlash' in rules):
            self.replaceSlash=True
        if ('flatEmail' in rules):
            self.flatEmail=True
        if ('flatPhone' in rules):
            self.flatPhone=True
        if ('flatYear' in rules):
            self.flatYear=True
        if ('flatMonth' in rules):
            self.flatMonth=True
        if ('flatNumber' in rules):
            self.flatNumber=True
        if ('lemmatize' in rules):
            self.lemmatize=True



    def isFolderEmpty(self,str):
        result=False
        for dirpath, dirnames, files in os.walk(str):
            if files:
                print('Not empty')
                result=False
            if not files:
                print('Empty')
                result=True

    def isEnglishWord_US(self, str):
        d=enchant.Dict("en_US")
        result=d.check(str)
        # print(str+' EnglishUS:',(result))
        return d.check(str)

    def isEnglishWord_GB(self, str):
        d=enchant.Dict("en_GB")
        result=d.check(str)
        # print(str+' EnglishGB:',(result))
        return d.check(str)


    def returnNonEnglishDictWords(self,strList=None):
        resultList=[]
        for item in strList:
            if (self.isEnglishWord_US(item)==False and self.isEnglishWord_GB(item)==False):
                resultList.append(item)
        return resultList

    def unifiedShuffle(self,npA, npB):
        """
        Shuffle numpyA and numpyB in the same order
        :param npA:
        :param npB:
        :return:
        """
        npA=np.array(npA)
        npB = np.array(npB)
        assert len(npA)==len(npB)
        p=np.random.permutation(len(npA))
        return npA[p],npB[p]

    def tokensToStr(self, tokens=None, delimiter=','):
        counter=0
        strToken=''
        for token in tokens:
            if(counter==0):
                strToken=str(token)
            else:
                strToken=strToken+delimiter+str(token)
            counter=counter+1
        return strToken

    def clearscreen(self):
        import os, sys
        sys.stdout.write(os.popen('clear').read())

    def readFileContent(self,filename):
        file=open(filename,'r')
        content=file.read()
        file.close()
        return content

    def compareCsrMatrix(self,matrix1, matrix2):
        """
        First check shape equality,
        then check num of nonzero elements
        :param matrix1:
        :param matrix2:
        :return:
        """
        results=False
        if(matrix1.shape==matrix2.shape):
            if (matrix1 - matrix2).nnz == 0:
                if(matrix1.sum()==matrix2.sum()):
                    results=True
        return results

    def profileObjSize(self,obj=None):
        metric='bytes'
        size=sys.getsizeof(obj)
        if(size>1000):
            size=size/1000
            metric='kb'
        elif(size>1000000):
            size=size/1000000
            metric='mb'
        elif(size>1000000000):
            size=size/1000000000
            metric='gb'
        return str(size)+ ' ' + metric

    def ifFileExists(self, filename):
        return os.path.exists(filename)

    def tokenize(self, rawStr, removeStopwords=True, toLowercase=True, replaceSlash=True, flatEmail=False, flatPhone=False, flatYear=False, flatMonth=False, flatNumber=False, lemmatize=False):
        myStr=rawStr
        # print(myStr)
        ## This area can only deal as raw string
        if (self.toLowercase==True):
            myStr = myStr.lower()

        if (self.replaceSlash==True):
            myStr=myStr.replace('/', ' ')

        if (self.lemmatize == True):

            # nltk.download('wordnet')
            results = []
            wordnet_lemmatizer = WordNetLemmatizer()
            myStrTokens = myStr.split(' ')
            for token in myStrTokens:
                results.append(wordnet_lemmatizer.lemmatize(token))
            myStr = self.tokensToStr(results, ' ')

        if (self.flatEmail==True):
            #Replace emails with a fixed representation
            myStr,_ = re.subn(r'\b[\w.-]+?@\w+?[\.\w+?\b]+',"WCNEMAIL" , myStr)

        if (self.flatNumber==True):
            # print(myStr)
            myStr = re.sub(r'\d', 'X', myStr)
            # print(myStr)

        if (self.flatPhone==True):
            #Break string into tokens seperated by spaces
            #For each word
                #Check if word falls into ascii 43,48-57, 123,125,45 only
                    #If yes
                        #If mark found ==True, delete this token
                        #If mark found ==False, set mark==True , change token to WCNPHONE, add to string
                    #If no
                        #If mark found ==True, set mark found==False, add to string
                        #If mark found ==False, add to string

            print('Original:', myStr)
            phoneASCIIA = [43, 45, 40, 41]
            phoneASCIIB = list(range(48, 58))
            phoneASCIIA.extend(phoneASCIIB)
            flatPhoneTokens = myStr.split(' ')
            adjustedStringTokens = []
            found = False
            for myToken in flatPhoneTokens:
                asciiList = [ord(myChar) for myChar in myToken]
                sizeOfIntersect = len(list(set(phoneASCIIA) & set(asciiList)))
                if (sizeOfIntersect == len(set(asciiList))):
                    # All Matched ASCII for phone
                    if (found == False):
                        found = True
                        adjustedStringTokens.append('WCNPHONE')
                else:
                    # Not a match for phone
                    if (found == True):
                        found = False
                        adjustedStringTokens.append(myToken)
                    else:
                        adjustedStringTokens.append(myToken)

            myStr = self.tokensToStr(adjustedStringTokens, ' ')



        ## This area can only deal after tokenized.
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(myStr)

        if (self.removeStopwords==True):
            en_stop = get_stop_words('en')
            stopped_tokens = [i for i in tokens if not i in en_stop]
            tokens=stopped_tokens

        if (self.flatYear==True):
            resultTokens=[]
            for token in tokens:
                if self.isYear(token):
                    resultTokens.append('WCNYEAR')
                else:
                    resultTokens.append(token)
            tokens=resultTokens

        if (self.flatMonth==True):
            resultTokens=[]
            for token in tokens:
                if self.isMonth(token):
                    resultTokens.append('WCNMONTH')
                else:
                    resultTokens.append(token)
            tokens=resultTokens


        # print(tokens)
        return tokens


    def isMonth(self,str):
        result=False
        regex=r"jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december"
        match=re.match(regex,str.lower())
        if(match!=None):
            result=True
        return result

    def isYear(self, str):
        result=False
        try:
            strToInt=int(str)
            if(strToInt>1900 and strToInt<2050):
                result=True
        except Exception:
            result=False
        return result

    def setupLogFileLoc(self,logFile):
        """
        This has to be applied if using logDebug method.
        :param logFile:
        :return:
        """
        print('Setting log file to ',logFile)

        self._logFile=logFile
        self.logger = logging.getLogger('Utility')
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        filelogger = logging.FileHandler(logFile)
        filelogger.setFormatter(formatter)
        self.logger.addHandler(filelogger)

    def logInfo(self, owner, errMsg):
        if (self._logFile!=''):
            fullmsg=owner+':'+errMsg+'\n'
            self.logger.info(self.COLOR_GREEN+fullmsg+self.COLOR_NORMAL)
        else:
            print("Error: No log file indicated. Use setupLogFileLoc(filename) to init")

    def logError(self, owner, errMsg):
        if (self._logFile!=''):
            fullmsg = owner + ':' + errMsg + '\n'
            self.logger.error(self.COLOR_RED+fullmsg+self.COLOR_NORMAL)
        else:
            print("Error: No log file indicated. Use setupLogFileLoc(filename) to init")

    def logDebug(self, owner, errMsg):
        """
        Log a message into stdout and file.
        :param owner:
        :param errMsg:
        :return:
        """
        if (self._logFile!=''):
            fullmsg = owner + ':' + errMsg + '\n'
            self.logger.debug(fullmsg)
        else:
            print("Error: No log file indicated. Use setupLogFileLoc(filename) to init")

    starttime=None
    def startTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This must be the first method to call before calling stopTimeTrack or checkpointTimeTrack
        Will start the recording of time.
        :return:
        """
        self.starttime=datetime.datetime.now()
        pass

    def stopTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This is called in pair with startTimeTrack everytime.
        It will print time lapse after startTimeTrack
        :return:
        """
        endtime=datetime.datetime.now()
        duration=endtime-self.starttime
        result = ""

        m, s = divmod(duration.seconds, 60)
        h, m = divmod(m, 60)
        result = "Time taken: " + str(h) + " hours, " + str(m) + ' mins and ' + str(s) + 'secs'
        return result
        pass

    def getDatetime(self):
        return (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    def checkpointTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This can be called consecutively for as many times as long as startTimeTrack has been first called.
        It will print the time lapse from last check point
        E.g.
        Utility().startTimeTrack)_
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        :return:
        """
        endtime = datetime.datetime.now()
        duration = endtime - self.starttime
        result=""
        m, s = divmod(duration.seconds, 60)
        h, m = divmod(m, 60)
        result = "Time taken: " + str(h) + " hours, " + str(m) + ' mins and ' + str(s) + 'secs'
        self.starttime=endtime
        # print(result)
        return result
        pass

    def getFileNameFromFullFilePath(self,fullfilepath):
        return fullfilepath.split('/')[-1]

    def getFullPathFromFilename(self,filename):
        tokens=filename.split('/')
        path=''
        for folder in tokens[:-1]:
            path = path+folder+'/'
        return path

    def getFileTypeFromFullFilePath(self,fullfilepath):
        return fullfilepath.split('/')[-1].split('.')[-1]

    def getFileNameWithoutTypeFromFullFilePath(self,fullfilepath):
        return fullfilepath.split('/')[-1].split('.')[0]

    def dumpObjToFile(self,obj=None, filename=None):
        joblib.dump(obj,filename)

    def loadFileToObj(self,filename=None):
        return joblib.load(filename)

    def removeWhiteSpaces(self,aString):
        return (''.join(aString.split()))

    def replaceWhiteSpaceWith(self,aString, replacementStr):
        return aString.replace(' ', replacementStr)

    def strip_non_ascii(self,string):
        ''' Returns the string without non ASCII characters'''
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)

    def isContentEnglish(self, someStr):
        """
        Check content of someStr for English language
        :param someStr:
        :return:
        """
        isEng=False
        DetectorFactory.seed = 0
        try:
            if langdetect.detect(someStr)=='en':
                isEng=True
        except Exception as error:
            isEng=False
        return isEng

    def recreateDir(self,folder):
        print('Checking destination folder exists... ')
        if os.path.exists(folder):
            # print('This is a destructive action, rmtree ' + folder + '?\n please confirm with Y')
            if (True):
                if os.path.isdir(folder):
                    print('recreateDir','Destination folder exists... Removing')
                    shutil.rmtree(folder)
                else:
                    print('recreateDir','Destination exists as file... Removing')
                    os.remove(folder)
            else:
                print('recreateDir', 'User aborted')
                exit (-1)
        os.makedirs(folder)
        print('recreateDir', 'Recreated...')

    def pathfinder(self,type,filename,prefix='',ext=False):
        """
        This function will build the path to the required CV file or App json
        :param type: TYPE_CV or TYPE_APP
        :param filename: The filename that you want to build the path for
        :param prefix: Any prefix for the folder path
        :return: Full path to the path.
        """
        finalPath=None
        if type==self.TYPE_CV:
            filenameTokens=filename.split('_')
            clientid=filenameTokens[2].zfill(3)
            if (ext):
                clientid=clientid+'-ext'
            hashid=filenameTokens[-1].split('.')[0]
            fileExt=filenameTokens[-1].split('.')[-1]
            hashidCharList=list(hashid)
            folderPath=hashidCharList[0]+'/'+hashidCharList[0]+hashidCharList[1]+'/'+hashidCharList[0]+hashidCharList[1]+hashidCharList[2]+'/'+hashidCharList[0]+hashidCharList[1]+hashidCharList[2]+hashidCharList[3]+'/'+hashidCharList[0]+hashidCharList[1]+hashidCharList[2]+hashidCharList[3]+hashidCharList[4]
            finalPath=prefix+'/'+clientid+'/'+folderPath+'/'+filename

            if (os.path.exists(finalPath)==False):
                filenameTokens = filename.split('_')
                clientid = filenameTokens[2].zfill(3)
                clientid = clientid + '-ext'
                hashid = filenameTokens[-1].split('.')[0]
                fileExt = filenameTokens[-1].split('.')[-1]
                hashidCharList = list(hashid)
                folderPath = hashidCharList[0] + '/' + hashidCharList[0] + hashidCharList[1] + '/' + hashidCharList[0] + \
                             hashidCharList[1] + hashidCharList[2] + '/' + hashidCharList[0] + hashidCharList[1] + \
                             hashidCharList[2] + hashidCharList[3] + '/' + hashidCharList[0] + hashidCharList[1] + \
                             hashidCharList[2] + hashidCharList[3] + hashidCharList[4]
                finalPath = prefix + '/' + clientid + '/' + folderPath + '/' + filename

        elif type==self.TYPE_APP:
            filenameTokens = filename.split('_')
            clientid = filenameTokens[1].zfill(3)
            appid=filenameTokens[2].split('.')[0].zfill(2)
            appidList=list(appid)

            if len(appidList)>1:
                hashidCharList=appid[:-2].zfill(7)
                folderPath = hashidCharList[0] + '/' \
                             + hashidCharList[0] + hashidCharList[1] + '/' \
                             + hashidCharList[0] + hashidCharList[1] + hashidCharList[2] + '/' \
                             + hashidCharList[0] + hashidCharList[1] + hashidCharList[2] + hashidCharList[3] + '/' \
                             + hashidCharList[0] + hashidCharList[1] + hashidCharList[2] + hashidCharList[3] + hashidCharList[4]  + '/' \
                             + hashidCharList[0] + hashidCharList[1] + hashidCharList[2] + hashidCharList[3] + hashidCharList[4]+ hashidCharList[5]  + '/' \
                             + hashidCharList[0] + hashidCharList[1] + hashidCharList[2] + hashidCharList[3] + hashidCharList[4] + hashidCharList[5] + hashidCharList[6]
                finalPath=prefix + '/' + clientid.zfill(3)+'/'+folderPath+'/'+filename

        return finalPath

    def saveStringToFile(self, string,filename):
        f=open(filename,'w')
        f.write(str(string))
        f.close()

    def saveListToFile(self,myList, filename):
        f=open(filename,'w')
        for item in myList:
            f.write("%s\n" % item)
        f.close()

    def isUpperCase(self, character):
        result=False
        if (ord(character)>=65 and ord(character)<=90):
            result=True
        return result

    def isLF(self,character):
        result=False
        if (ord(character)==10 or ord(character)==13):
            result=True
        return result

    def getInfoFromAppD2vFilename(self,appD2Vfilename):
        """

        :param appD2Vfilename:
        :return:
        """
        filename=appD2Vfilename.split('/')[-1]
        tokens=filename.split('_')
        clientid=tokens[1]
        appid=tokens[2].split('.')[0]
        return clientid, appid

    def getInfoFromCVd2vFilename(self,cvD2Vfilename):
        """

        :param cvD2Vfilename:
        :return:
        """
        # file_private_32_0a80cb1a02da98fecfaa31f6a717393199cdd89d_32_1052275_008.cvd2v
        filename=cvD2Vfilename.split('/')[-1]
        tokens=filename.split('_')
        clientid=tokens[2]
        hashcode=tokens[3]
        appid=tokens[5]
        paraid=tokens[6].split('.')[0]
        return clientid,appid,hashcode,paraid

if __name__ == "__main__":
    util=Utility()
    tokens = util.tokenize(
    rawStr='This is a test to replace my month, Oct, or October,  my year 1979, 19791, my phone 8826492 and 989267840 email j@y123.234.com to a flatten value of another email jax79sg@yahoo.com.sg',
    flatYear=True, flatEmail=True,flatPhone=True,flatMonth=True)


# u=Utility()
# print(u.pathfinder(u.TYPE_CV,'file_private_32_a52eaac52cd896c5dd1308f4e2dd04f548abed07.txt','/home/kah1'))
# print(u.pathfinder(u.TYPE_APP,'app_032_5.json'))



# print("Reading feature list")
# all_df=pd.read_csv('../data/features_full_plusnouns_pluspuidthresh.csv')
# feature_train_df = all_df[:74067]
# featureDF=Utility().artificialFeatureExtension(feature_train_df)
# featureDF.to_csv('nonlinear_features')
# print("Created featureDF: ", list(featureDF))

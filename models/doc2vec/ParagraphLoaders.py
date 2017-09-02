#DOC2VEC

import gensim
import os
from gensim.models.doc2vec import TaggedDocument
from random import shuffle
fileext='d2v'


def get_doc_list(folder_name):
    """
    Credits: https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/
    For loading all txt files under a directory, it returns a list of strings. If you have 4 txts, then the length of the list will be 4.
    :param folder_name:
    :return:
    """
    doc_list = []
    file_list = [folder_name + '/' + name for name in os.listdir(folder_name) if name.endswith(fileext)]
    for file in file_list:
        st = open(file, 'r').read()
        doc_list.append(st)
    print('Found %s documents under the dir %s .....' % (len(file_list), folder_name))
    return doc_list, file_list

def _limitSize(docList, maxSize):
    """
    Reduce the document size down to required number.
    :param docList:
    :param maxSize:
    :return:
    """
    finalSize=maxSize
    if len(docList)<maxSize:
        finalSize=len(docList)
    return docList[:finalSize]

def get_doc(folder_name, maxdocs=18000, util=None):
    """
    Credits: https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/
    customise to recognisesub folders
    :param folder_name:
    :return:
    """
    # util=Utilities.Utility()
    # util.setupLogFileLoc('/u01/bigdata/doc2vecframe.log')
    doccounter=0

    taggeddoc = []
    # texts = []

    util.logDebug(util.PARALOADER,
                         'Gathering file information from ' + folder_name + '...this will take a while')
    noOfFiles = sum([len(files) for r, d, files in os.walk(folder_name)])
    util.logDebug(util.PARALOADER,
                         'No of files found: ' + str(noOfFiles) + '\nProcessing now')

    # subfolders=[x[0] for x in os.walk(folder_name)]
    # for subfolderpath in subfolders[1:len(subfolders)]:
    subfolderpath=folder_name
    subfolder=subfolderpath.split('/')[-1]
    util.logDebug(util.PARALOADER,
                  'Getting documents under: ' + subfolder)
    doc_list, file_list = get_doc_list(subfolderpath)
    # tokenizer = RegexpTokenizer(r'\w+')
    # en_stop = get_stop_words('en')
    # p_stemmer = PorterStemmer()

    for index, i in enumerate(doc_list):
        if(doccounter%10000==0):
            util.logDebug(util.PARALOADER, 'Created ' + str(doccounter) + ' tagged documents of ' + str(noOfFiles) + ' raw docs')
        # for tagged doc
        wordslist = []
        tagslist = []

        # clean and tokenize document string
        # raw = i.lower()
        stopped_tokens=util.tokenize(i)
        #Convert '/' to spaces
        # raw = raw.replace('/',' ')
        # tokens = tokenizer.tokenize(raw)


        # remove stop words from tokens
        # stopped_tokens = [i for i in tokens if not i in en_stop]
        # util.logDebug('PARAGRAPHLOADERS','No of stop words removed: '+str(len(tokens)-len(stopped_tokens)))

        # # remove numbers
        # number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        # number_tokens = ' '.join(number_tokens).split()
        #
        # # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # remove empty
        # length_tokens = [i for i in tokens if len(i) > 1]       #This line is a must
        # add tokens to list
        # texts.append(length_tokens)
        tag=subfolder+'_'+str(file_list[index].split('/')[-1])
        # print("tags: " + tag )
        # print(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split())
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stopped_tokens))).split(), [tag])
        taggeddoc.append(td)
        doccounter=doccounter+1
    subfolder=None
    doc_list=None
    tokenizer=None
    util=None
    doccounter=None
    # texts=None
    noOfFiles=None
    subfolders=None
    taggeddoc=_limitSize(taggeddoc,maxdocs)


    # fileObject = open(folder_name + '/cvUsed.csv', 'a')

    shuffle(taggeddoc)
    return taggeddoc
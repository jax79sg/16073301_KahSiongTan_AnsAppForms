Kah Siong's Thesis: Learning to Answer Recruitment Application Forms. 
==============================

This repository only contain the source codes. No data or saved models will be published here as they are confidential. You may use the following structure to request the relevant data from the industrial partner.


All references to file structures points to Server SP3, unless otherwise stated.

Source Code Structure (Except for src folder, all other folders were not included in the public repository)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── alldata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   └── ToLabel    
    │   │       └── ForAgreementTest  <- Contains the 10 identical CVs rated by individual. Used for Kappa compute.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Not in use
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Not in use
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- WIP thesis and generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate interim and processed data
    │   │   └── docx2text  <- Scripts to convert docx files to text files
    │   │   └── extractFromAppJSON  <- Scripts to extract from application form raw data and generate interim data.
    │   │   └── extractFromCV   <- Scripts to extract from CV converted text data and generate interim data
    │   │   └── pdf2text    <- Scripts to convert PDF to text 
    │   │   └── vxCVtypes   <- Scripts to compile information on CV data
    │   │   └── vxStates    <- Scripts to compule information on the states  on application forms.
    │   │
    │   ├── features       <- Not used
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── doc2vec     <- Scripts that train doc2vec models from processed data and infer new vectors
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------
File System on SP3 Structure(/home/kah1)
------------

    ├── remote_cookie_runtime   <- A runtime replica of Project Code Structure
    ├── backup_u01         
    │   ├── *.zip.bz2  <- Backups of the most important files    
    ├── nltk_data          <- Needed to perform lemmatization.
    ├── vX                 <- Local copy of selected data from BigData server
    ├── software           <- Installers for extra software needed.
    ├── requirementsSP3.txt           <- PIP requirements file 
--------
File System on SP3 Structure(/u01/bigdata) - SSD disk, i used this for faster performance. 
------------
  
    ├── 00_appcvref        <- Output of DataPrepPhases 1 and 2
    ├── 00_appjson         <- Contains selected application forms from BigData server
    ├── 00_cv_text   
    │   └── CV2            <- Contains the converted text of selected CVs
    ├── 01a_d2v_QandA      <- Output of Data Phase 3
    ├── 01b_d2v            <- Output of Data Phase 4
    ├── 02d_d2vModel1      <- Storage location of all Vector Space Models and Topic Models.
    │   └── significance           
    │   │   └── stuart     <- Storage directory of results from Stuart significance testing.
    │   │   │   └── *.predictions     <- Part of results from prediction models, used to compute Stuart values
    │   │   │   └── *.smpvalue        <- Pvalues of Stuart Significance Testing
    │   │   └── wilcoxon   <- Storage directory of results from Wilcoxon significance testing.
    │   │   │   └── *.F1   <- Part of results from prediction models, used to compute Stuart values
    │   │   │   └── *.wilpvalue        <- Pvalues of Wilcoxon Significance Testing
    │   └── results           
    │   │   └── *.metric     <- Results from prediction models. .
    │   │   └── *.CONFU     <- Confusion Matrices from prediction models. .
    │   │   └── *.predictions     <- Prediction results from prediction models.
    │   │   └── *.png     <- Confusion matrix (image) from prediction models.
    │   │   └── *.summary     <- Summary of *.metrics 
    │   │   └── *.eval     <- log file.
    │   └── features       <- Working directory of 
    │   │                        1. All features (E.g. Application forms and balanced CV sections) in their vector formats. 
    │   │                        2. All fitted models of Machine Learning models
    │   │   └── original   <- Working directory of unbalanced CV sections in their vector formats.
    │   └── featureset3NoPreproc       <- Same structure as features, except that this is based on the set of features from client 032 and no special preprecessing.    
    │   └── featureset5client200NoProc <- Same structure as features, except that this is based on the set of features from client 200 and no special preprecessing.
    │   └── featureset7client219NoProc <- Same structure as features, except that this is based on the set of features from client 219 and no special preprecessing.    
    │   └── featureset6client32dataprocessed <- Same structure as features, except that this is based on the set of features from client 032 and with data flattening and lemmatization.    
    ├── 03a_01b_test            <- Output of Data Phase 8
    ├── 03b_raters              <- Output of "Creating new unseen testset by labelling" part 3

--------
Legend
------------
- Done - Means full run completed with data generated and stored
- Pending - Means coding completed pending a run.
- Running - Means the run is in progress.
- Coding - Means coding in progress
- Yet to code - Means coding has yet to start
- XX refers to the client id. (E.g. 032, 048, 051)
- YYYYY refers to the app id.(E.g. 12314, 25345)
- ZZZ refers to the category (E.g. EDU, SKILLS, PERSONALDETAILS).
- PP refers to the Nth paragraph

Phases and their code references
------------

- End2End Pipeline Demo
    - Use End2EndPipelineDemo.py in the pipeline package to launch.
    - Take the raw CVs in SP3://u01/bigdata/02d_d2vModel1/testCV/*.* and convert into text
    - Split the text
    - Load the vector model
    - Load the prediction model
    - Convert split text to vectors
    - Perform prediction
    - Save results

Overview: Basically for each CV, extract into portions. And then infer the possible category (EDU, SKILLS, PERSONAL DETAILS or WORK EXPERIENCE)
for each portion. 
- Data Prep phase 0. (docx2txt.sh and pdf2text.sh) - Done for all
    - Use DATAPREPPhase0.sh in the pipeline package to launch
    - Convert PDF and DOCX CVs into text formats
    - Converted results stored alongside the file it converted.

- Data Prep phase 0a.(vxCVtypes.py) - Done for 32,48,51, 200
    - This phase is purely for information gathering, not needed.
    - set directoryToProbe, directoryToSave
    - Scans directoryToProbe (/home/vX/CV/) and build a full table of CVs and its full paths.
    - Save results in format of (cfilename,clientid,filetype,fullpath)
    - Results stored in directoryToSave(bigdata:/home/kah1/vX/cvtypes/) 

- Data Prep phase 0b. (insert_cv_ref_to_mysql.py - insertcvtypes method)
    - This phase is a one time operation which was completed for all clients (no need to do again)
    - It simply loaded the  /vX/CV/XX/files_XX.csv content into MySQL (cv_ref table) for some processing.    
    
- Data Prep phase 1. (insert_cv_ref_to_mysql.py) - Done 32,48,51
    - Use DATAPREPPhase1.py in the pipeline package to launch
    - Traverse the cv_ref table in mysql to allow us to;
        - Check for file exists and if conversion successful (E.g. English and Non-garbage)
        - Link up Cvs and AppIDs and OppIDs..
    - Save in format ('cv_ref_id', 'appid', 'candidateid', 'oppid', 'cfilename', 'clientid', 'isfileexists', 'iscontentenglish',
        'fullfilename')
    - Results stored in SP3:/u01/bigdata/00_appcvref/cv_ref_XX.csv

- Data Prep phase 2. (refineUpdateCVRefCSV.py) - Done
    - Use DATAPREPPhase2.py in the pipeline package to launch
    - Filter the cv_ref_xx.csv in data phase 1 to exclude non-english and failed conversions
    - Save in format ('cv_ref_id', 'appid', 'candidateid', 'oppid', 'cfilename', 'clientid', 'isfileexists', 'iscontentenglish',
        'fullfilename')
    - Results stored in SP3:/u01/bigdata/00_appcvref/cv_ref_XX_filtered.csv.


- Data Prep phase 3. (appToDoc2vec.py) - Done for 32,48,51
    - Use DATAPREPPhase3.py in the pipeline package to launch
    - For the app json in SP3:/home/kah1/vX/apps_opps/app/XX/*.json, break it down into paragraphs by consolidating section_title
    as the broad question/category, and the specific questio3ns and answers below it as the sentences. (Look at a SP3:/home/kah1/vX/apps_opps/app/XX/*.json file and you will understand)
    - The extracted portions will be placed in respective folders named by the section_title.
    - Results stored in /u01/bigdata/01a_d2v_QandA/**

- Data Prep phase 4. (Manual) - Done for 32 only (Focus on Client 32..baseline)
    - Manually (Human) organise folders of sp3://u01/bigdata/01a_d2v_QandA/ such that categories are grouped into SKILLS, EDU,PERSONALDETAILS and WORK EXPERIENCE
    - Results stored in sp3://u01/bigdata/01b_d2v/032/edu/doc2vecEdu/* sp3://u01/bigdata/01b_d2v/032/skills/doc2vecSkills/* and sp3://u01/bigdata/01b_d2v/032/personaldetails/doc2vecPersonalDetails/* and sp3://u01/bigdata/01b_d2v/032/workexp/doc2vecWorkexp/*

- Data Prep phase 5. (SummaryAppD2V.py - createSummary()) - Done
    - Use DATAPREPPhase5.py in the pipeline package to launch
    - Build a table of valid ZZZ,XX,YYYYY , by scanning through SP3:/u01/bigdata/01b_d2v/ZZZ/app_XX_YYYYY.d2v
    - Results stored in /u01/bigdata/01b_d2v/032/edu/summary_32_edu.csv /u01/bigdata/01b_d2v/032/skills/summary_32_skills.csv and /u01/bigdata/01b_d2v/032/personaldetails/summary_32_personaldetails.csv

- Data Prep phase 6. (CV_CategoryInference.py - joinCVREF_APPD2V()) - Done
    - Use DATAPREPPhase6.py in the pipeline package to launch
    - So by now...The valid document app data is held in /u01/bigdata/01b_d2v/ZZZ/app_XX_YYYYY.d2v
    - The only valid CVs to use for eval are those that correspond to the XX and YYYYY above. These corresponding CV can be found
    in /u01/bigdata/00_appcvref/cv_ref_XX_filtered.csv. which has a mapping of XX, YYYY to a correct CV path.
    - Join phase 5 table (/u01/bigdata/01b_d2v/032/edu/summary_32_edu.csv) with Dataphrase2(/u01/bigdata/00_appcvref/cv_ref_XX_filtered.csv),with clientid and appid as join key.
    - So we will have a mapping of valid app section (d2v extension file) identified by category, clientid, appid, to a CV path held in cv_ref_XX_filtered.csv.
    - Join /u01/bigdata/00_appcvref/cv_ref_32_filtered.csv with /u01/bigdata/01b_d2v/32/edu/summary_32_edu.csv
    - Join /u01/bigdata/00_appcvref/cv_ref_32_filtered.csv with /u01/bigdata/01b_d2v/32/skills/summary_32_skills.csv
    - Join /u01/bigdata/00_appcvref/cv_ref_32_filtered.csv with /u01/bigdata/01b_d2v/32/personaldetails/summary_32_personaldetails.csv
    - Results stored in /u01/bigdata/01b_d2v/032/edu/joind2vSummaryCVpath.csv, /u01/bigdata/01b_d2v/032/skills/joind2vSummaryCVpath.csv, /u01/bigdata/01b_d2v/032/personaldetails/joind2vSummaryCVpath.csv

- Data Prep phase 7.(CV_CategoryInference.py - splitTrainTestset) - Done
    - Use DATAPREPPhase7.py in the pipeline package to launch
    - We now have all the relevant data. Now we should split the data into train and test sets.
    - Given results from phase 6, break into 80% train/20% eval for each category ZZZ.
    - Save the 80% and 20% into different folders for APPd2v and CV.
    - Results stored in
        - /u01/bigdata/01b_d2v/032/edu/doc2vecEdu/train, /u01/bigdata/01b_d2v/032/edu/doc2vecEdu/test
        - /u01/bigdata/01b_d2v/032/skills/doc2vecSkills/train, /u01/bigdata/01b_d2v/032/skills/doc2vecSkills/test
        - /u01/bigdata/01b_d2v/032/personaldetails/doc2vecpersonaldetails/train, /u01/bigdata/01b_d2v/032/personaldetails/doc2vecpersonaldetails/test

- Data Prep phase 8. (CVtoDoc2vec.py-extractCVunderDataset) - Running
    - Use DATAPREPPhase8.py in the pipeline package to launch
    - Now that the training set is defined, we can start to extract the CVs sections in bulk and use them for subsequent training.
    - For each category ZZZ
        - For each /u01/bigdata/01b_d2v/032/ZZZ/trainset.csv and /u01/bigdata/01b_d2v/032/ZZZ/testset.csv .
            - Merge all edu, skill, personaldetails and work experience into one table and remove duplicates.
            - Find the CV for each row and perform an extraction of each CV.
                - Save extracted CV Section into corresponding CVd2v (cvfilename_XX_YYYYY_PP.cvd2v) files into /u01/bigdata/03a_01b_test/cvd2v/032/train/* and /u01/bigdata/03a_01b_test/cvd2v/032/test/*.
    - Results stored in /u01/bigdata/03a_01b_test/cvd2v/032/train/*.cvd2v and /u01/bigdata/03a_01b_test/cvd2v/032/test/*.cvd2v
    - Note: d2v does not mean Doc2Vec, its just a file extension used to indicate extracted sections.

- Vector Space Model Phase Bag Of Words (BOW.py)
    - Use VSMPhase_BOW.py in the pipeline package to launch
    - This one builds Vector Representation based on Bag Of Words
        - This means the sentence can be represented by a series of numbers.
    - Training corpus is based on extracted CV sections.
    - The generated vector space model is saved in /u01/bigdata/02d_d2vModel1/cvBowVectorSpaceModel_*****.model
        - ***** refers to the number of dimensions

- Vector Space Model Phase Term Frequency Inverse Document Frequency (TFIDF.py)
    - Use VSMPhase_TFIDF.py in the pipeline package to launch
    - This one builds Vector Representation based on Term Frequency Inverse Document Frequency
        - This means the sentence can be represented by a series of numbers.    
    - Training corpus is based on extracted CV sections.
    - The generated vector space model is saved in /u01/bigdata/02d_d2vModel1/cvTfidfVectorSpaceModel_*****.model
        - ***** refers to the number of dimensions

- Vector Space Model Phase Paragraph2Vec (W2V.py)
    - Use VSMPhase_W2V.py in the pipeline package to launch
    - This one builds Vector Representation based on Word2Vec
        - This means the sentence can be represented by a series of numbers.    
    - Training corpus is based on extracted CV sections.
    - The generated vector space model is saved in /u01/bigdata/02d_d2vModel1/cvW2v***VectorSpaceModel.model
        - *** refers to the number of dimensions    

- Vector Space Model Phase Paragraph2Vec (D2V.py)
    - Use VSMPhase_D2V.py in the pipeline package to launch
    - This one builds Vector Representation based on Paragraph2Vec
        - This means the sentence can be represented by a series of numbers.    
    - Training corpus is based on extracted CV sections.
    - The generated vector space model is saved in /u01/bigdata/02d_d2vModel1/cvD2v***VectorSpaceModel.model
        - *** refers to the number of dimensions    

- Topic Modelling Model Phase (LDA.py)
    - Use VSMPhase_LDA.py in the pipeline package to launch
    - This one builds the LDA topic models based on a predefined basket size.
    - Training corpus is based on extracted CV sections.
    - The generated vector space model is saved in /u01/bigdata/02d_d2vModel1/CvLda4TopicModel.model
    
- Model Train Phase
    - There is no need to perform any specific steps for model training. 
    - This is integrated into the respective models. 
    - During evaluation, the models are automatically trained and saved if existing saved models cannot be found.
    - See ModelPhase_MLCLASSIFIER, ModelPhase_SIMILARITYMODEL and ModelPhase_TOPICMODEL for launch points
    
- Build Feature phase (build_features.py)
    - Use BatchBuildAllFeatures.sh in the pipeline package to launch
    - With the Vector Space Models and the appD2v and cvD2v files, it is not possible to create a set of features for appD2v and cvD2v.
    - The features are basically represented in format of (filename,content,label,vector)
    - These features can then be used in any machine learning or deep learning applications.
    - All features saved in /u01/bigdata/02d_d2vModel1/features/*.features  
 
- Adhoc Build new CVs features 
    - Use BatchCreateFeaturesFromCVs.sh in the pipeline package to launch
    - If there's a need to test new CVs, use this launcher.
    - Given the information on the CV (text) location and Vector Space model info
        - Break the CVs into sections
        - Generate a set of features that can be tested on the prediction models

- Creating new unseen testset by labelling
    1. From a set of CVs, create a set of data for human labelling 
        - Use PREPARE_CVSectionForLabelling in the pipeline package to launch
        - Note: All CVs sections are not labelled. Thus, to perform evaluation, at least some CV sections must be labelled.
        - Given a folder containing all the CV sections (cvd2v files)
        - This step will go though and generate a set of CVs for use with the labelling software.
        - Results stored in '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/test/CV_200_Extracts_XX.csv
    2. Label the CVs
        - Put the contents of '/u01/bigdata/03a_01b_test/cvd2v/200/labelled/test/CV_200_Extracts_XX.csv in data folder of label app (E.g. /home/kah1/remote_cookie_runtime/src/data/LabelApp/data).
        - Run the Label App and perform labelling (/home/kah1/remote_cookie_runtime/src/data/LabelApp/main/py to launch labelapp)
        - Labelled data stored in users folder of label app (E.g. /home/kah1/remote_cookie_runtime/src/data/LabelApp/users)
    3. Perform agreement between labellers and get final labels
        - Use BuildRatersLabelledcvd2v.py in the pipeline package to launch
        - Results stored in /u01/bigdata/03b_raters/client200/majorityVotedRatersCvd2v/0,1,2,3
    4. Build the features based on this labels
        - E.g. python3 /home/kah1/remote_cookie_runtime/src/features/build_features.py '/u01/bigdata/02d_d2vModel1/features/log_cvTrainW2v300min1features.log' '/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/0,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/1,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/2,/u01/bigdata/03b_raters/majorityVotedRatersCvd2v/3' '0,1,2,3' 'w2v' '/u01/bigdata/02d_d2vModel1/cvW2v300VectorSpaceModel.model' '/u01/bigdata/02d_d2vModel1/features/cvTrainW2v300min1.features'
        - Results stored in /u01/bigdata/02d_d2vModel1/features/cvTrainW2v300min1.features in this case
    5. Balance the dataset (So its easier to observe)
        - Use commons.scratchpad.scratch.py's balanceTest() method
        - Results stored as *.feature2 (You can save the orginals elsewhere)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

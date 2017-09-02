#' Stuart Maxwell Test Statistic
#' Objective :
#'   To compute pairwise StuartMaxwell statistics for a set of predictions
#'   1. Find all *.predictions files in the given prediction folder
#'   2. Create a pairwise set for all prediction files
#'   3. Load prediction file in pairwise manner
#'     a. Each prediction file expected to have format label,prediction as columns
#'   4. Create agreement table based on the 2 sets of predictions
#'   5. Run Stuart Maxwell on the agreement table
#'   6. Return Test Statistic, Degree Of Freedom and pValue based on Chi-Square Distribution
#'   7. Optional: Save results to file, named in pairwise manner.
#' Usage:
#'   1. stuartmaxwell_pairwise(predictionFolder = 'Full folder path')
#'   2. Results will be saved in *.sm and *.smpvalue files
#' Created by: kah siong
#' Created on: 05/08/17
#' @author Kah Siong, Tan
#' @example stuartmaxwell_pairwise(predictionFolder = '/home/user/predictionfolder/')
#' @param A full path to the folder containing all the prediction files
#' @return Print out and saved files
#' @export

stuartmaxwell_pairwise = function (predictionFolder)
{
  #Checking and installing dependancies
  list.of.packages <- c("coin")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)

  #Loading all prediction files.
  files=list.files(path=predictionFolder,pattern='*predictions')

  #Create pairwise predictions
  pairwiseFilenames=combn(files,2)
  print(pairwiseFilenames)
  print(dim(pairwiseFilenames))
  noOfPairs=dim(pairwiseFilenames)[2]

  #Processing pairwise
  for (i in 1:noOfPairs)
  {
    file1=pairwiseFilenames[1,i]
    file2=pairwiseFilenames[2,i]
    print(cat(c('\n\n\nProcessing ',file1,' and ',file2,"\n")))
    #print(file1)
    #print(file2)
    #print('\n')
    prediction1<-read.csv(paste(predictionFolder,file1,sep=''), header=TRUE,sep=",")
    prediction2<-read.csv(paste(predictionFolder,file2,sep=''), header=TRUE,sep=",")
    print(prediction1$prediction)
    print(prediction2$prediction)

    #Create agreement table
    twowaytable=table(prediction1$prediction,prediction2$prediction)
    #print(twowaytable)
    print("Agreement table")
    print(addmargins((twowaytable)))
    library(coin)

    mhresult=NULL
    df=NULL
    q=NULL
    pvalue=NULL

    trycatchresult=tryCatch({
      #Run Stuart Maxwell test
      mhresult=(mh_test(twowaytable, distribution=approximate(B=9999)))
      df=mhresult@statistic@df
      q=(mhresult@statistic@teststatistic)
      pvalue=pchisq(q,df,lower.tail=FALSE)

      #Pack results
      results=''
      results=paste('Stuart Maxwell Test Statistic:',q,sep=' ')
      results=paste(results,'Degree Of Freedom:',sep='\n')
      results=paste(results,df,sep=' ')
      results=paste(results,'pValue:',sep='\n')
      results=paste(results,pvalue,sep=' ')
      pvalueString=''
      if(pvalue<=0.05)
      {
        pvalueString=  'P<=0.05 - Reject null hypothesis. Therefore significant difference'
      }
      else
      {
        pvalueString=  'P>0.05 - Cannot reject null hypothesis. Therefore NO significant difference'
      }
      results=paste(results,pvalueString,sep='\n')
    }, warning=function(w){
      print('WARNING: MH_TEST FAILED.. likely a class is not predicted at all')
      print(w)
      results='Cannot compute, likely a class is not predicted at all'
      pvalue=100
    }, error=function(e){
      print('ERROR: MH_TEST FAILED.. likely a class is not predicted at all')
      print(e)
      results='Cannot compute, likely a class is not predicted at all'
      pvalue=100
    }, finally = {

    })

    #Print and save results
    print(cat(results))
    file1approach=(strsplit(file1,"[.]"))[[1]]
    file2approach=strsplit(file2,"[.]")[[1]]
    finalFile=paste(file1approach[1],file2approach[1],sep='-')
    finalFile=paste(finalFile,'.sm',sep='')
    finalFile=paste(predictionFolder,finalFile,sep='')

    pvalueFile=paste(file1approach[1],file2approach[1],sep='-')
    pvalueFile=paste(pvalueFile,'.smpvalue',sep='')
    pvalueFile=paste(predictionFolder,pvalueFile,sep='')
    print(c('Saving summary to ',finalFile))
    sink(finalFile)
    cat(results)
    sink()
    print(c('Saving pvalue to ',pvalueFile))
    sink(pvalueFile)
    cat(pvalue)
    sink()

  }
}

#stuartmaxwell_pairwise(predictionFolder = '/home/kah1/u01/bigdata/02d_d2vModel1/features/')

#' Wilcoxon Test Statistic
#' Objective :
#'   To compute pairwise Wilcoxon statistics for a set of predictions
#'   1. Find all *.F1 files in the given prediction folder
#'   2. Create a pairwise set for all F1 files
#'   3. Load F1 file in pairwise manner
#'   5. Run Wilxocon on the pair
#'   6. Return Test Statistic and pValue.
#'   7. Optional: Save results to file, named in pairwise manner.
#' Usage:
#'   1. wilcoxon_pairwise(scoresFolder = 'Full folder path')
#'   2. Results will be saved in *.wil and *.wilpvalue files
#' Created by: kah siong
#' Created on: 07/08/17
#' @author Kah Siong, Tan
#' @example wilcoxon_pairwise(scoresFolder = '/home/user/predictionfolder/')
#' @param A full path to the folder containing all the F1 files
#' @return Print out and saved files
#' @export

wilcoxon_pairwise = function (scoresFolder)
{
scoresFolder= '/home/kah1/u01/bigdata/02d_d2vModel1/features/'
  #Checking and installing dependancies
  list.of.packages <- c("coin")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)

  #Loading all prediction files.
  files=list.files(path=scoresFolder,pattern='\\.F1')
  print(files)
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
    score1<-read.csv(paste(scoresFolder,file1,sep=''), header=F,sep=",")
    score2<-read.csv(paste(scoresFolder,file2,sep=''), header=F,sep=",")
    print(score1$V1)
    print(score2$V1)

    library(coin)

    mhresult=NULL
    df=NULL
    q=NULL
    pvalue=NULL

    trycatchresult=tryCatch({
      #Run Wilcoxon
      mhresult=wilcox.test(score1$V1,score2$V1,exact=T)
      pvalue=mhresult$p.value
      q=mhresult$statistic
      #Pack results
      results=''
      results=paste('Wilcoxon Test Statistic:',q,sep=' ')
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
    finalFile=paste(finalFile,'.wil',sep='')
    finalFile=paste(scoresFolder,finalFile,sep='')

    pvalueFile=paste(file1approach[1],file2approach[1],sep='-')
    pvalueFile=paste(pvalueFile,'.wilpvalue',sep='')
    pvalueFile=paste(scoresFolder,pvalueFile,sep='')
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

#wilcoxon_pairwise(scoresFolder = '/home/kah1/u01/bigdata/02d_d2vModel1/features/')

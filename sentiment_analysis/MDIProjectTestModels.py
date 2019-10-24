# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:04:39 2019

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle 

from MDIProjectSentimentAnalysis import (mdiProjectExtractFeatures, mdiProjectGetFeatureVectorForSingleTweet, 
                                         mdiProjectProcessTweet, mdiProjectStopWordList, mdiProjectReadFiles, mdiProjectReadSampleFile,
                                         mdiProjectSplitTrainTest)
def naiveBayesClassifierTestModel():
    
    mdiProjectSampleTweetList = mdiProjectReadSampleFile(mdiProjectReadFiles("mdiProjectFiles","SampleTrainingData.csv"))
    mdiProjectTrainingSet, mdiProjectTestingSet = mdiProjectSplitTrainTest(mdiProjectSampleTweetList)
    mdiProjectStopWordsList = mdiProjectStopWordList(mdiProjectReadFiles("mdiProjectFiles","StopWords.txt"))
    
    #mdiProjectTestingSet = mdiProjectTestingSet[1:10]
    
    mdiProjectActualSentiments = []
    mdiProjectPredictedSentiments = []
    
    #Open NaiveBayesClassifier picke file
    with open("NaiveBayesClassifierModel.pkl","rb") as NaiveBayesClassifier:
        NaiveBayesClassifierModel = pickle.load(NaiveBayesClassifier)
        
    for mdiProjectTestingTweetRow in mdiProjectTestingSet:
        
        mdiProjectSentiment = mdiProjectTestingTweetRow[0]
        mdiProjectTweet = mdiProjectTestingTweetRow[1]
    
        #Process the tweet
        porcessedTweet = mdiProjectProcessTweet(mdiProjectTweet)
        
        #Get the features vector for a single tweet
        featureVector = mdiProjectGetFeatureVectorForSingleTweet(porcessedTweet, mdiProjectStopWordsList)
        
        #Get the feacture words
        featureWords = mdiProjectExtractFeatures(featureVector)
        
        #Get sentiments based on the feature words
        tweetSentiment = NaiveBayesClassifierModel.classify(featureWords)
        
        mdiProjectActualSentiments.append(mdiProjectSentiment)
        mdiProjectPredictedSentiments.append(tweetSentiment)
    
    return mdiProjectActualSentiments, mdiProjectPredictedSentiments

def naiveBayesClassifierTestModelResult():
    
    mdiProjectActualSentiments, mdiProjectPredictedSentiments = naiveBayesClassifierTestModel()
    #print(mdiProjectActualSentiments)
    #print(mdiProjectPredictedSentiments)

    for actualSentimentPos, actualSentiment in enumerate(mdiProjectActualSentiments):
        
        if(actualSentiment == "positive"):
            
            mdiProjectActualSentiments[actualSentimentPos] = 1
        else:
            mdiProjectActualSentiments[actualSentimentPos] = 0
    
    for predictedSentimentPos, predictedSentiment in enumerate(mdiProjectPredictedSentiments):
        
        if(predictedSentiment == "positive"):
            
            mdiProjectPredictedSentiments[predictedSentimentPos] = 1
        else:
            mdiProjectPredictedSentiments[predictedSentimentPos] = 0
            
    #print(mdiProjectActualSentiments)
    #print(mdiProjectPredictedSentiments)
    
    result = confusion_matrix(mdiProjectActualSentiments, mdiProjectPredictedSentiments)
    print(result)
    
    accuracy = accuracy_score(mdiProjectActualSentiments, mdiProjectPredictedSentiments)
    print(accuracy)
    
    print(classification_report(mdiProjectActualSentiments, mdiProjectPredictedSentiments))
    
def logisticRegressionClassifierTestModel():
    
    
    with open("mdiProjectLogisticRegressionclassifier.pkl","rb") as mdiProjectLogisticRegressionclassifier:
        mdiProjectLogisticRegressionclassifierModel = pickle.load(mdiProjectLogisticRegressionclassifier)
    
    
    with open("mdiProjectLogisticRegressionX_test.pkl","rb") as mdiProjectLogisticRegressionX_test:
        X_test = pickle.load(mdiProjectLogisticRegressionX_test)
    
    with open("mdiProjectLogisticRegressionY_test.pkl","rb") as mdiProjectLogisticRegressionY_test:
        y_test = pickle.load(mdiProjectLogisticRegressionY_test)
    
    # Testing model performance
    mdiProjectSentimentPredict = mdiProjectLogisticRegressionclassifierModel.predict(X_test)

    mdiProjectLogisticRegressionCnfusionMatrix = confusion_matrix(y_test, mdiProjectSentimentPredict)
    print(mdiProjectLogisticRegressionCnfusionMatrix)
    
    print(classification_report(y_test, mdiProjectSentimentPredict))  
    
    accuracy = accuracy_score(y_test, mdiProjectSentimentPredict)
    print(accuracy)
    
if __name__ == "__main__":
    logisticRegressionClassifierTestModel()
    
#8/8/18

#text classifier script for classifying the type of wine variety based its description.
import pandas as pd 
import numpy as np 
from io import StringIO
import matplotlib.pyplot as plt
import pylab 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

sample_df=pd.read_csv("winemag-data_first150k.csv",encoding="latin-1")

class Analyzer():
    def __init__(self):
        pass
    def text_classifier(self,df):
        sample_df1=sample_df.iloc[0:500]
        sample_df1=sample_df[["description","variety"]]
        #encode response (output) as an integer
        sample_df1['variety_id']=sample_df1['variety'].factorize()[0]
        #tf-idf
        tfidf=TfidfVectorizer(sublinear_tf=True,min_df=5,norm='l2',encoding='latin-1',ngram_range=(1,3),stop_words='english')
        features=tfidf.fit_transform(sample_df1.description).toarray()
        labels=sample_df1.variety_id
        #multi-class classifier (naive bayes)
        X_train,X_test,y_train,y_test=train_test_split(sample_df1['description'],sample_df1['variety'],random_state=0)
        count_vect=CountVectorizer() 
        X_train_counts=count_vect.fit_transform(X_train) #sparse matrix 
        tfidf_transformer=TfidfTransformer()
        X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts) #tfidf score for each vector description
        classifier=MultinomialNB().fit(X_train_tfidf,y_train)
        #predict the class output 
        classifier.predict(count_vect.transform(['Mac Watson honors the memory of a wine once made by his mother in this tremendously delicious, balanced and complex botrytised white. Dark gold in color, it layers toasted hazelnut, pear compote and orange peel flavors, reveling in the succulence of its 122 g/L of residual sugar.']))
        #accuracy evaluations (random forest model)
        model_rf=RandomForestClassifier(n_estimators=150,max_depth=2)
        fit_rf=model_rf.fit(features,labels)
        accuracy=cross_val_score(fit_rf,features,labels,scoring="accuracy",cv=3)
        acc_mean=accuracy.mean()  
        return acc_mean 

if __name__ == '__main__':
    text_classifier = Analyzer()
    text_classifier.text_classifier(sample_df)



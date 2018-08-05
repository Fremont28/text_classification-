#8/5/18 
#text classification-predicting the wine variety based on the wine description using tf-idf 
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

grape=pd.read_csv("winemag-data_first150k.csv",encoding="latin-1")
grape.head(3)
grape['variety'].unique() #unique wine varieties 
grape.info()

grape1=grape[0:2000]
grape1_sub=grape1[["description","variety"]]
grape1_sub 

#check for non-null rows 
grape1_sub.isnull().values.any() 

#encode variety as integer 
grape1_sub['variety_id']=grape1_sub['variety'].factorize()[0]


#tf-idf
tfidf=TfidfVectorizer(sublinear_tf=True,min_df=5,norm='l2',encoding='latin-1',ngram_range=(1,3),stop_words='english')

features=tfidf.fit_transform(grape1_sub.description).toarray()
labels=grape1_sub.variety_id
features.shape 

#multi-class classifier (naive bayes)
X_train,X_test,y_train,y_test=train_test_split(grape1_sub['description'],grape1_sub['variety'],
random_state=0)

count_vect=CountVectorizer() 
X_train_counts=count_vect.fit_transform(X_train) #sparse matrix (1500x4786) string to vector repr. 
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts) #tfidf score for each vector description?

classifier=MultinomialNB().fit(X_train_tfidf,y_train)

grape1_sub.iloc[2]["description"]
grape1_sub.iloc[2]["variety"]

#predictions
classifier.predict(count_vect.transform(['Mac Watson honors the memory of a wine once made by his mother in this tremendously delicious, balanced and complex botrytised white. Dark gold in color, it layers toasted hazelnut, pear compote and orange peel flavors, reveling in the succulence of its 122 g/L of residual sugar.']))

#accuracy evaluations 
model_rf=RandomForestClassifier(n_estimators=150,max_depth=2)
fit_rf=model_rf.fit(features,labels)

accuracy=cross_val_score(fit_rf,features,labels,scoring="accuracy",cv=3)
accuracy.mean()  


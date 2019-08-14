# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:41:53 2019

@author: sathvik.inti
"""

import pandas as pd
import os
import json
import io 
from pprint import pprint
from sklearn_pandas import DataFrameMapper, cross_val_score
from pandas.compat import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir(r'C:\Users\sathvik.inti\Desktop\Techgig_hack2')
train=pd.read_csv('UserIdToGender_Train.csv')

#train1=train.head(400)
test=pd.read_csv('UserId_Test.csv')

os.chdir(r'C:\Users\sathvik.inti\Desktop\Techgig_hack2\Urls_Json_Data') 
Urls_Json_Data =[]
for line in open('Urls_data.txt', 'r',encoding="utf8"):
    Urls_Json_Data.append(json.loads(line))

Urls_Json_Data1=pd.DataFrame(Urls_Json_Data)
url_data11=Urls_Json_Data1.head(900)

os.chdir(r'C:\Users\sathvik.inti\Desktop\Techgig_hack2\UserIdToUrl')
files=os.listdir()
main_df=pd.DataFrame()
for i in files:
    with open(i) as f:
        contents =f.read()
    df = pd.read_fwf(io.StringIO(contents))
    df['userid'], df['url'] = df['userid,url'].str.split(',', 1).str
    df=df.drop(['userid,url'],axis=1)
    main_df=main_df.append(df)
        
##dask
#import dask.dataframe as dd
#d = dd.from_pandas(main_df1, 2)
#d1=d['userid,url'].str.split(",")
#d1=d['userid,url'].str.split(",")
#d1=d1.to_frame()
#d1 = d1.assign(left=d1.lists.map(lambda x: x[0]),
#                 right=d1.lists.map(lambda x: x[1]))
#df=d1.compute()
#

main_df1=main_df.head(300)
#train['userid'].isna().sum()
train['userid']=train['userid'].astype(str)
#main_df.dropna(inplace=True)
main_df['userid']=main_df['userid'].astype(str)
main_df=pd.DataFrame(main_df.groupby(['userid'])['url'])
main_df.columns=['userid','url']
main_df['userid']=main_df['userid'].astype(str)

train_v1=pd.merge(train,main_df,left_on="userid",right_on="userid",how='inner')
#train_v1.to_csv('Train_URL.csv')
train_v1.isna().sum()
#train_v2=train_v1.dropna()
train_v1['gender']=[0 if train_v1['gender'][i]=='M' else 1 for i in range(len(train_v1))]
pd.value_counts(train_v1['gender'])
#train_vv=train_v1.head(300)

train_v1['url1']=train_v1['url'].apply(' '.join).str.replace('[^A-Za-z\s]+', '').str.split(expand=False)
train_v1=train_v1.drop('url',axis=1)
train_v1['url'] = [','.join(map(str, l)) for l in train_v1['url1']]
train_v1=train_v1.drop('url1',axis=1)

#%%
#EDA
from nltk.probability import FreqDist
fdist = FreqDist(train_v1[train_v1['gender']==1]['url'])
top_ten = fdist.most_common(10)
print(top_ten)

#%%

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
x = tfidfconverter.fit_transform(train_v1['url'].values)
#train_v2['tweetsVect']=list(x)
#train_v2=train_v2.drop('url1',axis=1)
#X=train_v2[['userid',  'tweetsVect']]
Y=train_v1[['gender']].values
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, max_depth=2,random_state=551552,class_weight={'M':1,'F':4})
#class_weight="balanced"
rf_model=clf.fit(x,Y)

from sklearn import svm
svm_classifier = svm.SVC(gamma='scale',class_weight={'F': 10})
svm_classifier.fit(x,Y)

import pickle
filename = 'svm_1.sav'
pickle.dump(svm_classifier, open(filename, 'wb'))
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)

test['userid']=test['userid'].astype(str)
testv1=pd.merge(test,main_df,left_on="userid",right_on="userid",how='left')
testv1.isna().sum()
#testv2=testv1.dropna()
testv1['url1']=testv1['url'].apply(' '.join).str.replace('[^A-Za-z\s]+', '').str.split(expand=False)
testv1=testv1.drop('url',axis=1)
testv1['url'] = [','.join(map(str, l)) for l in testv1['url1']]
testv1=testv1.drop('url1',axis=1)

x_test = tfidfconverter.fit_transform(testv1['url'].values)
preds=rf_model.predict(x_test)
preds_svm=svm_classifier.predict(x_test)

test['gender']=preds
pd.value_counts(test['gender'])
test.to_csv('sub1_rf2.csv',index=False)















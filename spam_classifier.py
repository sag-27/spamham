# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:25:19 2023

@author: Sagnik Sen
"""

import numpy as np
import pandas as pd
sms=pd.read_csv("C:/Users/Sagnik Sen/Downloads/sms+spam+collection/SMSSpamCollection",sep='\t',names=["label","message"])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()

clean=[]
for i in range(len(sms)):
    msg=re.sub('[^a-zA-Z]',' ',sms['message'][i])
    msg=msg.lower()
    msg=msg.split()
    msg=[lem.lemmatize(word) for word in msg if word not in set(nltk.corpus.stopwords.words('english'))]
    msg=' '.join(msg)
    clean.append(msg)
    
#bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
bow=cv.fit_transform(clean).toarray()

find=list(zip(*np.where(bow==1)))

y=pd.get_dummies(sms['label'])
y=y.iloc[:,1].values            # spam==1 ham==0

#train_test_split
from sklearn.model_selection import train_test_split as tts
message_train,message_test,label_train,label_test=tts(bow,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(message_train,label_train)

y_pred=spam_detect_model.predict(message_test)

from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(label_test,y_pred)

from sklearn.metrics import accuracy_score
acc= accuracy_score(label_test,y_pred)

#ROCAUC plot
import sklearn
fpr, tpr, threshold = sklearn.metrics.roc_curve(label_test, y_pred)
roc_auc = sklearn.metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'green', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--',color='blue')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.fill_between(fpr, tpr,color='black',alpha=0.2)
plt.grid(color='black')
plt.show()
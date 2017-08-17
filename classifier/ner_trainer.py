import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn import svm, model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

df=pd.read_csv("../dataset/data_final_inflado_tag.csv")
tags=df['tags'].tolist()[1:]
# print(list(enumerate(tags)))
words=df['words'].tolist()[1:]
# print(words)

def prepara_frase(tags,words):
    features=[]
    feature={}
    targets=[]
    for ind,tag in enumerate(tags):
        if tag!='-':
            feature['0']=words[ind-3]
            feature['1']=words[ind-2]
            feature['2']=words[ind-1]
            feature['3']=words[ind]
            feature['4']=words[ind+1]
            feature['5']=words[ind+2]
            feature['6']=words[ind+3]
            features.append(feature)
            targets.append(tag)
    return features,targets

features,targets=prepara_frase(tags,words)
vectorizer = DictVectorizer(sparse=False)
vectorizer.fit(features)
joblib.dump(vectorizer, 'vectorizer_entity.pkl')


transformed=vectorizer.transform(features)
# print(features)
# print(targets)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(transformed, targets, test_size=0.1, random_state=42) #Se intenta quitar por deprecated warning
X_train, X_test, y_train, y_test = model_selection.train_test_split(transformed, targets, test_size=0.10, random_state=42)

# print(np.isnan(X_train).any())
# np.isnan(y_train).any()
# np.isnan(X_test).any()
# np.isnan(y_test).any()
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
# print(y_test)
lin_svc = svm.LinearSVC(C=1).fit(X_train, y_train)
# svm_machine = svm.SVC(probability=True).fit(X_train, y_train)
# print(str(accuracy_score(y_test, lin_svc.predict(X_test))))
# joblib.dump(lin_svc, 'clasifier_entity.pkl')
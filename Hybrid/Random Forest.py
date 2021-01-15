import pandas as pd
import numpy as np

import os, psutil
pid = os.getpid()
ps = psutil.Process(pid)
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


# # Feature Extraction
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
# selector = SelectKBest(f_classif, k=100)
# selected_features = selector.fit_transform(featureMatrix, labelVector)
#
# print((-selector.scores_).argsort()[:])
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, verbose=3)
startrainTime = datetime.now()
clf.fit(featureMatrixTR, labelVectorTR)
tt1 =time()
TrainingTime =tt0-tt1
import pickle
filename = 'RandomF_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



## TESTING

from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = loaded_model.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits =4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, Y_pred.round())
print('ROC AUC: %f' % auc)

memUse = ps.memory_full_info()
print(memUse)

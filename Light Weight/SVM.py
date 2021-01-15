import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

from sklearn.svm import SVC

## MODEL

classifier = SVC(kernel='rbf', random_state = 1, verbose=3)

def training():
    classifier.fit(featureMatrixTR,labelVectorTR)

classifier.fit(featureMatrixTR,labelVectorTR)

import pickle
filename = 'SVM_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

def testing():
    loaded_model.predict(featureMatrix)

## TESTING
Y_pred = classifier.predict(featureMatrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labelVector,Y_pred)
accuracy = float(cm.diagonal().sum())/len(labelVector)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix
SVM_predictions = classifier.predict(featureMatrix)
cm = confusion_matrix(labelVector, SVM_predictions)
print(classification_report(labelVector, SVM_predictions,digits =4))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, SVM_predictions)
print('ROC AUC: %f' % auc)

import timeit
import os
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat('SVM_model.sav').st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)

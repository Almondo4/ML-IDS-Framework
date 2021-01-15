import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

from sklearn.naive_bayes import GaussianNB
## MODEL

gnb = GaussianNB()


#Train the model using the training sets
def training():
    gnb.fit(featureMatrixTR, labelVectorTR)

gnb.fit(featureMatrixTR, labelVectorTR)


import pickle
filename = 'NAIVB_model.sav'
pickle.dump(gnb, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

#Predict the response for test dataset
def testing():
    loaded_model.predict(featureMatrix)

y_pred = gnb.predict(featureMatrix)
# Model Accuracy, how often is the classifier correct?
## TESTING
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(labelVector, y_pred, digits=4))
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(labelVector, y_pred)
print('ROC AUC: %f' % auc)


import timeit
import os
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat('NAIVB_model.sav').st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)



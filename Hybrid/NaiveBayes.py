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
# Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(featureMatrixTR, labelVectorTR)

#Predict the response for test dataset
y_pred = gnb.predict(featureMatrix)
# Model Accuracy, how often is the classifier correct?
## TESTING
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(labelVector, y_pred, digits=4))
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(labelVector, y_pred)
print('ROC AUC: %f' % auc)


import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../Data/Dynamic_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()

model.fit(featureMatrixTR, labelVectorTR)

predictions = model.predict(featureMatrix)

## TESTING

from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(labelVector, predictions)
print(classification_report(labelVector, predictions))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, predictions.round())
print('ROC AUC: %f' % auc)

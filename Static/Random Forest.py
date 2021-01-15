import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Static_Training.csv")
DataTest = pd.read_pickle("../Data/Static_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet

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

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, verbose=3)

clf.fit(featureMatrixTR, labelVectorTR)


## TESTING
Y_pred = clf.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = clf.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits = 4))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions.round())
print('ROC AUC: %f' % auc)


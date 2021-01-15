import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/CICMalDroid_Train.csv")
DataTest = pd.read_pickle("../../Data/CICMalDroid_Test.csv")
# DataZero = pd.read_pickle("../../Data/ZD.csv")



featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
# featureMatrixZ = DataZero.iloc[:,:-1].values
# labelVectorZ = DataZero.iloc[:,-1].values


#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)
# featureMatrixZ = sc.fit_transform(featureMatrixZ)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)
# labelVectorZ = labelencoder.fit_transform(labelVectorZ)

# # Feature Extraction
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
# selector = SelectKBest(f_classif, k=100)
# selected_features = selector.fit_transform(featureMatrix, labelVector)
#
# print((-selector.scores_).argsort()[:])

import xgboost as xgb
train = xgb.DMatrix(featureMatrixTR, label=labelVectorTR)
test = xgb.DMatrix(featureMatrix, label=labelVector)
# testZ = xgb.DMatrix(featureMatrixZ, label=labelVectorZ)

param = {
    "max_depth":4,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 13,
    "verbosity": 3}
epochs = 3

model = xgb.train(param, train, epochs)
predictions = model.predict(test)


## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = model.predict(test)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labelVector)
labelVector = lb.transform(labelVector)
RF_predictions = lb.transform(RF_predictions)

auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
# auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo", labels=["0","1","2","3","4","5","6","7","8","9","10","11","12"] )
print('ROC AUC: %f' % auc)


# ZeroDay Tests

# from sklearn.metrics import classification_report,confusion_matrix
# predictionZ = model.predict(testZ)
# cm = confusion_matrix(labelVectorZ, predictionZ)
# print(classification_report(labelVectorZ, predictionZ,digits=4))
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(labelVectorZ, predictionZ.round())
# print('ROC AUC: %f' % auc)


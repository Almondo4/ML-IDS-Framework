import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/AndMal_Zero_Train.csv")
DataTest = pd.read_pickle("../../Data/AndMal_Zero_Test.csv")
DataZero = pd.read_pickle("../../Data/AndMal_Zero_Day.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
featureMatrixZ = DataZero.iloc[:,:-1].values
labelVectorZ = DataZero.iloc[:,-1].values



#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)
featureMatrixZ = sc.fit_transform(featureMatrixZ)

labelVectorTR= labelVectorTR.astype(np.int32)
labelVector= labelVector.astype(np.int32)
labelVectorZ= labelVectorZ.astype(np.int32)

import xgboost as xgb
train = xgb.DMatrix(featureMatrixTR, label=labelVectorTR)
test = xgb.DMatrix(featureMatrix, label=labelVector)
testZ = xgb.DMatrix(featureMatrixZ, label=labelVectorZ)
import catboost as cb
cb = cb.CatBoostClassifier(depth=8,learning_rate=1,iterations=600,task_type="GPU",devices='0:1')

cb.fit(featureMatrixTR, labelVectorTR)



## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = cb.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions)
print('ROC AUC: %f' % auc)


# # Zero Day

print("############################################################ ZERO DAY")

from sklearn.metrics import classification_report
RF_predictions = cb.predict(featureMatrixZ)
print(classification_report(labelVectorZ, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVectorZ,RF_predictions)
print('ROC AUC: %f' % auc)

from tensorflow import keras
import pandas as pd

dynamicModel = keras.models.load_model("SAE_CICMAl.h5")
dynamicModel = dynamicModel.layers[0]



DataTrain = pd.read_pickle("../../../Data/CICMalDroid_Train.csv")
DataTest = pd.read_pickle("../../../Data/CICMalDroid_Test.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)



featureMatrixTR = dynamicModel.predict(featureMatrixTR)
featureMatrix =dynamicModel.predict(featureMatrix)

import xgboost as xgb
train = xgb.DMatrix(featureMatrixTR, label=labelVectorTR)
test = xgb.DMatrix(featureMatrix, label=labelVector)

param = {
    "max_depth":100,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 3}
epochs = 300

model = xgb.train(param, train, epochs)


# # Report
# from sklearn.metrics import classification_report
# model_predictions = model.predict(test)
# # cm = confusion_matrix(labelVector, model_predictions)
# print(classification_report(labelVector, model_predictions.round(),digits =4))
#
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(labelVector, model_predictions.round())
# print('ROC AUC: %f' % auc)

## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions =  model.predict(test)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labelVector)
labelVector = lb.transform(labelVector)
RF_predictions = lb.transform(RF_predictions)

auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
print('ROC AUC: %f' % auc)

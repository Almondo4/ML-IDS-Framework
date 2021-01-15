# TensorBoard

import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'


# ############### Command to  start TensorBoard

# DATA
# # Dynamic
import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../../Data/Dynamic_Testing.csv")

featureMatrixTRD = DataTrain.iloc[:,:-1].values
labelVectorTRD = DataTrain.iloc[:,-1].values
featureMatrixD = DataTest.iloc[:,:-1].values
labelVectorD = DataTest.iloc[:,-1].values

# # Static

DataTrain = pd.read_pickle("../../Data/Static_Training.csv")
DataTest = pd.read_pickle("../../Data/Static_Testing.csv")

featureMatrixTRS = DataTrain.iloc[:,:-1].values
labelVectorTRS = DataTrain.iloc[:,-1].values
featureMatrixS = DataTest.iloc[:,:-1].values
labelVectorS = DataTest.iloc[:,-1].values
# # Hybrid
DataTrain = pd.read_pickle("../../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../../Data/Hybrid_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# Models
from tensorflow import keras
import tensorflow as tf

staticModel = keras.models.load_model("SAE_DLS_2.h5")
staticModel = staticModel.layers[0]
dynamicModel = keras.models.load_model("SAE_DLD_2.h5")
dynamicModel = dynamicModel.layers[0]

# Creating Datasets
staticSet = staticModel.predict(featureMatrixTRS)
dynamicSet = dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)

staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, verbose=3)

clf.fit(DATATRaining, labelVectorTR)

predictions = clf.predict(DATATRaining)

## TESTING


XGB_predictions = clf.predict(DATATesting)

from sklearn.metrics import classification_report,confusion_matrix
# XGB_predictions_Classes =model.predict_classes(test)
#
cm = confusion_matrix(labelVector, XGB_predictions)
print(classification_report(labelVector, XGB_predictions))

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# XGB_predictions = XGB_predictions[:, 0]
# XGB_predictions_Classes = XGB_predictions_Classes[:, 0]


print(classification_report(labelVector, XGB_predictions))
# kappa = cohen_kappa_score(labelVector, XGB_predictions_Classes)
# print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(labelVector, XGB_predictions)
print('ROC AUC: %f' % auc)
# confusion matrix
# matrix = confusion_matrix(labelVector, XGB_predictions_Classes)
# print(matrix)



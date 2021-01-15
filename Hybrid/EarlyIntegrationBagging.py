# TensorBoard

import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'


# ############### Command to  start TensorBoard


# tensorboard --logdir=./my_logs --port=6006

#  DATA
import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


import tensorflow as tf
# Importing The Models
base_model = tf.keras.models.load_model("./SAE_DLHI_2.h5")
base_model = base_model.layers[0]

DATAtraining=base_model.predict(featureMatrixTR)
DATAtesting=base_model.predict(featureMatrix)


from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix as CM

model = BaggingClassifier(verbose=1)
model.fit(featureMatrixTR, labelVectorTR)
predictions = model.predict(featureMatrix)

## TESTING

from sklearn.metrics import classification_report,confusion_matrix
XGB_predictions = model.predict(featureMatrix)
# XGB_predictions_Classes =model.predict_classes(test)
#
cm = confusion_matrix(labelVector, XGB_predictions)
print(classification_report(labelVector, XGB_predictions))

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


print(classification_report(labelVector, XGB_predictions))
auc = roc_auc_score(labelVector, XGB_predictions)
print('ROC AUC: %f' % auc)




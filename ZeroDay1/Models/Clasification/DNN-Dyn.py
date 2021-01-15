import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'
#########################################################################
#########################################################################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
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



#  Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=1000, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.35))
model.add(keras.layers.Dense(units=1000, activation="relu", kernel_regularizer="l1_l2", name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=1000, activation="relu",  name="hiddenL_3"))
model.add(tf.keras.layers.Dropout(rate =0.45))
model.add(keras.layers.Dense(units=13, activation="sigmoid", kernel_regularizer="l1_l2",name="outLayer"))

# Compiling
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="sparse_categorical_crossentropy", metrics =["accuracy"])

# Training
cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(featureMatrixTR, labelVectorTR, batch_size=2048, epochs=100,
                    validation_split=0.2, callbacks = [tensorboard_cb])

## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = model.predict_classes(featureMatrix)

print(classification_report(labelVector, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions)
print('ROC AUC: %f' % auc)


# # Zero Day

print("############################################################ ZERO DAY")

from sklearn.metrics import classification_report
RF_predictions = model.predict_classes(featureMatrixZ)
print(classification_report(labelVectorZ, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVectorZ,RF_predictions)
print('ROC AUC: %f' % auc)

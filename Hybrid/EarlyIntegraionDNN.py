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

# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


import tensorflow as tf
# Importing The Models
base_model = tf.keras.models.load_model("./SAE_DLHI_2.h5")
base_model.trainable =False

# base_model= base_model.layers[0]
base_model._name = "EarlyIntegration_DL"
base_model = base_model.layers[0]


modelEncoder = tf.keras.Model(inputs=[base_model.input], outputs=[base_model.layers[0].output])



DNN = tf.keras.layers.Dense(units=100, activation="relu", name='input2model',input_shape=[100])(modelEncoder.output)
DNN= tf.keras.layers.Dropout(rate =0.35)(DNN)
DNN= tf.keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l1_l2', name='hiddenlayer')(DNN)
DNN= tf.keras.layers.Dropout(rate =0.35)(DNN)
DNN = tf.keras.layers.Dense(units=100, activation="relu", name='hiddenLayer2')(DNN)
DNN= tf.keras.layers.Dropout(rate =0.35)(DNN)
output = tf.keras.layers.Dense(units="1", activation="sigmoid", kernel_regularizer='l1_l2', name='mainOutput')(DNN)


# Combining Models
model = tf.keras.Model(inputs=[modelEncoder.input], outputs=[output])
#   Compiling Ann

model.compile(optimizer=tf.optimizers.Adam(),
            loss ="mse",
            metrics =["accuracy"]
            )
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
# Training
es = tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
cp = tf.keras.callbacks.ModelCheckpoint("EarlyIntegrationDNN.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(featureMatrixTR, labelVectorTR, batch_size=500, epochs=5000, validation_split=0.2,callbacks=[ tensorboard_cb,cp])




## TESTING

score = model.predict(featureMatrix)

from sklearn.metrics import classification_report,confusion_matrix
SAEDNN_predictions = model.predict(featureMatrix)
# SAEDNN_predictions_Classes =model.predict_classes(featureMatrix)
#
cm = confusion_matrix(labelVector, SAEDNN_predictions.round())
print(classification_report(labelVector, SAEDNN_predictions.round(),digits=4))

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

SAEDNN_predictions = SAEDNN_predictions[:, 0]
SAEDNN_predictions_Classes = SAEDNN_predictions_Classes[:, 0]

kappa = cohen_kappa_score(labelVector, SAEDNN_predictions_Classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(labelVector, SAEDNN_predictions)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(labelVector, SAEDNN_predictions_Classes)
print(matrix)


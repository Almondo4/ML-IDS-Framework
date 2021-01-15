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
DataTrain = pd.read_pickle("../../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../../Data/Dynamic_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


import tensorflow as tf

stacked_encoder = tf.keras.models.Sequential([
tf.keras.layers.Dense(input_shape=[len(featureMatrixTR[0])],units =60, name="SAE_DynamicLayer_1"),
tf.keras.layers.Dense(units=30, activation="relu", name="SAE_DynamicLayer_2_Features"),

])
stacked_decoder = tf.keras.models.Sequential([
tf.keras.layers.Dense(units=60, activation="relu", input_shape=[30], name="SAE_DynamicLayer_3"),
tf.keras.layers.Dense(units= len(featureMatrixTR[0]), activation="sigmoid",name="SAE_DynamicLayer_4"),
])
stacked_ae = tf.keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(loss="mse",
optimizer=tf.keras.optimizers.Adam(),
                   )

# Callbacks
cp = tf.keras.callbacks.ModelCheckpoint("SAE_DLD_2.h5",save_best_only =True, save_freq='epoch')
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# Training
# stacked_ae.build()
# stacked_ae.summary()

history = stacked_ae.fit(featureMatrixTR, featureMatrixTR,epochs=2, callbacks = [tensorboard_cb], validation_split=0.2)



## TESTING

# score = stacked_ae[0].predict(featureMatrix)
# score_2 = stacked_encoder.predict(featureMatrix)
# ali = stacked_ae.layers[0]
# score = ali.predict(featureMatrix)
#
# from sklearn.metrics import classification_report,confusion_matrixsk
# XGB_predictions = model.predict(test)
# XGB_predictions_Classes =model.predict_classes(test)
# #
# cm = confusion_matrix(labelVector, XGB_predictions)
# print(classification_report(labelVector, XGB_predictions))
#
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
#
# XGB_predictions = XGB_predictions[:, 0]
# XGB_predictions_Classes = XGB_predictions_Classes[:, 0]
#
# kappa = cohen_kappa_score(labelVector, XGB_predictions_Classes)
# print('Cohens kappa: %f' % kappa)
# # ROC AUC
# auc = roc_auc_score(labelVector, XGB_predictions)
# print('ROC AUC: %f' % auc)
# # confusion matrix
# matrix = confusion_matrix(labelVector, XGB_predictions_Classes)
# print(matrix)

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


import tensorflow as tf

stacked_encoder = tf.keras.models.Sequential([
tf.keras.layers.Dense(input_shape=[len(featureMatrixTR[0])],units =200, name="SAE_HybridLayer_1"),
tf.keras.layers.Dense(units=100, activation="relu", name="SAE_HybridLayer_2_Features"),

])
stacked_decoder = tf.keras.models.Sequential([
tf.keras.layers.Dense(units=200, activation="relu", input_shape=[100], name="SAE_HybridLayer_3"),
tf.keras.layers.Dense(units= len(featureMatrixTR[0]), activation="sigmoid",name="SAE_HybridLayer_4"),
])
stacked_ae = tf.keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(loss="mse",
optimizer=tf.keras.optimizers.SGD(lr =0.4),
                   )

# Callbacks
cp = tf.keras.callbacks.ModelCheckpoint("SAE_DLHI_2.h5",save_best_only =True, save_freq='epoch')
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# Training
# stacked_ae.build()
# stacked_ae.summary()

history = stacked_ae.fit(featureMatrixTR, featureMatrixTR,epochs=500, callbacks = [tensorboard_cb, cp], validation_split=0.2)



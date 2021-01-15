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
from tensorflow.keras.layers import Concatenate
import tensorflow as tf

staticModel = keras.models.load_model("SAE_DLS_2.h5")
dynamicModel = keras.models.load_model("SAE_DLD_2.h5")

staticModel.trainable = False
dynamicModel.trainable = False


staticModel = staticModel.layers[0]
staticModel._name ="sae_S"
dynamicModel = dynamicModel.layers[0]
dynamicModel._name = "sae_D"

# Staticmodel output = 70
# Dynamicmodel output = 30


out = keras.layers.concatenate([dynamicModel.output,staticModel.output])
modelEncoder = keras.Model(inputs=[dynamicModel.input,staticModel.input], outputs=[out])


# Combining using AutoEncoder
DNN = keras.layers.Dense(units=200, activation="relu", name='input2model_DNN', input_shape=[100])(modelEncoder.output)
DNN= tf.keras.layers.Dropout(rate =0.25)(DNN)
DNN= keras.layers.Dense(units=300, activation="relu", name='hiddenlayer_DNN')(DNN)
DNN= tf.keras.layers.Dropout(rate =0.25)(DNN)
DNN = keras.layers.Dense(units=400, activation="relu", name='hiddenLayer2_DNN')(DNN)
DNN= tf.keras.layers.Dropout(rate =0.5)(DNN)
output = keras.layers.Dense(units="1", activation="sigmoid", name='mainOutput')(DNN)

# Combining Models
model = keras.Model(inputs=[modelEncoder.input], outputs=[output])
#   Compiling DNN

model.compile(optimizer=tf.optimizers.Adam(lr = 0.03),
            loss ="binary_crossentropy",
            metrics =["accuracy"]

            )

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='Multi SAE.png')

# Training
es = tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
cp = tf.keras.callbacks.ModelCheckpoint("IntermediateFusion2_.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit([featureMatrixTRD,featureMatrixTRS], labelVectorTR, batch_size=256, epochs=1500, validation_split=0.2,callbacks=[cp, tensorboard_cb])

# Testing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
DNN_predictions = model.predict([featureMatrixD,featureMatrixS])
cm = confusion_matrix(labelVector, DNN_predictions.round())
print(classification_report(labelVector, DNN_predictions.round()))

# ROC AUC
auc = roc_auc_score(labelVector, DNN_predictions.round())
print('ROC AUC: %f' % auc)

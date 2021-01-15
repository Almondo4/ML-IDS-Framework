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
dynamicModel = keras.models.load_model("SAE_DLD_2.h5")

# Creating Datasets
staticSet=staticModel.predict(featureMatrixTRS)
dynamicSet=dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)


# DNN = keras.layers.Dense(units=200, activation="relu", name='input2model_DNN', input_shape=[len(featureMatrix[0])])
# DNN= tf.keras.layers.Dropout(rate =0.25)(DNN)
# DNN= keras.layers.Dense(units=300, activation="relu", name='hiddenlayer_DNN')(DNN)
# DNN= tf.keras.layers.Dropout(rate =0.25)(DNN)
# DNN = keras.layers.Dense(units=400, activation="relu", name='hiddenLayer2_DNN')(DNN)
# DNN= tf.keras.layers.Dropout(rate =0.5)(DNN)
# output = keras.layers.Dense(units="1", activation="sigmoid", name='mainOutput')(DNN)

model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(DATATRaining[0])], units=100, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=100, activation="relu", name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=100, activation="relu", name="hiddenL_3"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=1, activation="sigmoid", name="outLayer"))
# Combining Models
# model = keras.Model(inputs=[modelEncoder.input], outputs=[output])
#   Compiling DNN

model.compile(optimizer=tf.optimizers.Adam(),
            loss ="binary_crossentropy",
            metrics =["accuracy"]

            )

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='Multi SAE.png')

# Training
es = tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
cp = tf.keras.callbacks.ModelCheckpoint("Combined_MultiSAE_2.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(DATATRaining, labelVectorTR, batch_size=521, epochs=4,
                    validation_split=0.2,callbacks=[cp, tensorboard_cb])

# Testing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)


DNN_predictions = model.predict(DATATesting)
cm = confusion_matrix(labelVector, DNN_predictions.round())
print(classification_report(labelVector, DNN_predictions.round()))

# ROC AUC
auc = roc_auc_score(labelVector, DNN_predictions.round())
print('ROC AUC: %f' % auc)



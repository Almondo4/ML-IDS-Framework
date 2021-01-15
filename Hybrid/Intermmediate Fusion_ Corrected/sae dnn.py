
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


staticModel = keras.models.load_model("../Late Integration/SAE_DLS_2.h5")
staticModel = staticModel.layers[0]
dynamicModel = keras.models.load_model("../Late Integration/SAE_DLD_2.h5")
dynamicModel = dynamicModel.layers[0]
fusion = keras.models.load_model("InterFusor.h5")
fusion = fusion.layers[0]

# Creating Datasets
# # Training
staticSet=staticModel.predict(featureMatrixTRS)
dynamicSet=dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)
DATATRaining = fusion.predict(DATATRaining)

# # Testing
staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)
DATATesting = fusion.predict(DATATesting)



#  Model Building
import  tensorflow as tf
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(DATATRaining[0])], units=100, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer="l1_l2", name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=100, activation="relu",  name="hiddenL_3"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer="l1_l2", name="outLayer"))



# Compiling
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="binary_crossentropy", metrics =["accuracy"])

# Training

# tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(DATATRaining, labelVectorTR, batch_size=1024, epochs=1000,
                    validation_split=0.2,)

# Testing
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
DNN_predictions = model.predict(DATATesting)
# cm = confusion_matrix(labelVector, DNN_predictions.round())
print(classification_report(labelVector, DNN_predictions.round(),digits=4))

# ROC AUC
auc = roc_auc_score(labelVector, DNN_predictions.round())
print('ROC AUC: %f' % auc)

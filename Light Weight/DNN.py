import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd

DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


import tensorflow as tf
from tensorflow import keras
import pandas as pd



featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


#  Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=100, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.25))
model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l1', name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.25))
model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l2', name="hiddenL_3"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer='l1_l2', name="outLayer"))

# Compiling
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="binary_crossentropy", metrics =["accuracy"])

# Training

# tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


def training():
    model.fit(featureMatrixTR, labelVectorTR, batch_size=2048, epochs=1000,
                    validation_split=0.2)

cp = tf.keras.callbacks.ModelCheckpoint("DNN.h5",save_best_only =True, save_freq='epoch')
history = model.fit(featureMatrixTR, labelVectorTR, batch_size=2048, epochs=1000,
                    validation_split=0.2, callbacks=[cp])

loaded_model = tf.keras.models.load_model("DNN.h5")

# Report

def testing():
    loaded_model.predict(featureMatrix)


model_predictions = model.predict(featureMatrix)


from sklearn.metrics import classification_report,confusion_matrix
DNN_predictions = model.predict(featureMatrix)
DNN_predictions_Classes =model.predict_classes(featureMatrix)
#
cm = confusion_matrix(labelVector, DNN_predictions.round())
print(classification_report(labelVector, DNN_predictions.round(),digits=4))

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

DNN_predictions = DNN_predictions[:, 0]
DNN_predictions_Classes = DNN_predictions_Classes[:, 0]

kappa = cohen_kappa_score(labelVector, DNN_predictions_Classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(labelVector, DNN_predictions.round())
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(labelVector, DNN_predictions_Classes)
print(matrix)

import timeit
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat("DNN.h5").st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)


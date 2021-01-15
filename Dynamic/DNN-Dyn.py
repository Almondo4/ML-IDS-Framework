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

DataTrain = pd.read_pickle("../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../Data/Dynamic_Testing.csv")


import tensorflow as tf
from tensorflow import keras
import pandas as pd



featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


#  Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=100, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.35))
model.add(keras.layers.Dense(units=100, activation="relu",  name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=100, activation="relu",  name="hiddenL_3"))
model.add(tf.keras.layers.Dropout(rate =0.45))
model.add(keras.layers.Dense(units=1, activation="sigmoid", name="outLayer"))

# Compiling
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="binary_crossentropy", metrics =["accuracy"])

# Training
cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(featureMatrixTR, labelVectorTR, batch_size=256, epochs=100,
                    validation_split=0.2, callbacks = [tensorboard_cb])

# Report
from sklearn.metrics import classification_report
model_predictions = model.predict(featureMatrix)
# cm = confusion_matrix(labelVector, model_predictions)
print(classification_report(labelVector, model_predictions.round(),digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, model_predictions.round())
print('ROC AUC: %f' % auc)

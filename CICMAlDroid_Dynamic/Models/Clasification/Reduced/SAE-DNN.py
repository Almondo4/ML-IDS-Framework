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
import tensorflow as tf
from tensorflow import keras

dynamicModel = keras.models.load_model("SAE_CICMAl.h5")
dynamicModel = dynamicModel.layers[0]



DataTrain = pd.read_pickle("../../../Data/CICMalDroid_Train.csv")
DataTest = pd.read_pickle("../../../Data/CICMalDroid_Test.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)



featureMatrixTR = dynamicModel.predict(featureMatrixTR)
featureMatrix =dynamicModel.predict(featureMatrix)


#  Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=200, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=200, activation="relu", kernel_regularizer="l1_l2", name="hiddenL_2"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=200, activation="relu",  name="hiddenL_3"))
model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=5, activation="sigmoid", kernel_regularizer="l1_l2", name="outLayer"))

# Compiling
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="sparse_categorical_crossentropy", metrics =["accuracy"])

# Training
cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit(featureMatrixTR, labelVectorTR, batch_size=1024, epochs=1000,
                    validation_split=0.2, callbacks = [tensorboard_cb])


## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = model.predict_classes(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labelVector)
labelVector = lb.transform(labelVector)
RF_predictions = lb.transform(RF_predictions)

auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
print('ROC AUC: %f' % auc)

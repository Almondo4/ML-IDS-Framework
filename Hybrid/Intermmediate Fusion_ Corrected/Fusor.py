
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

# Creating Datasets
# # Training
staticSet=staticModel.predict(featureMatrixTRS)
dynamicSet=dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)
# # Testing
staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

import tensorflow as tf
encoder = tf.keras.models.Sequential([tf.keras.layers.Dense(50,name ="Fusior_Encoder", input_shape=[len(DATATRaining[0])])])
decoder = tf.keras.models.Sequential([tf.keras.layers.Dense(len(DATATRaining[0]),name ="Fusior_Decoder", input_shape=[50])])
AE = tf.keras.models.Sequential([encoder, decoder])

AE.compile(loss="mse",
           optimizer=tf.keras.optimizers.Adam(),
           )

cp = tf.keras.callbacks.ModelCheckpoint("InterFusor.h5",save_best_only =True, save_freq='epoch')
history = AE.fit(DATATRaining, DATATRaining, epochs=1500, callbacks = [cp],validation_split = 0.2)

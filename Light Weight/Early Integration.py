import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
cp = tf.keras.callbacks.ModelCheckpoint("EarlyFusion.h5",save_best_only =True, save_freq='epoch')


# Training
# stacked_ae.build()
# stacked_ae.summary()


def training():
    stacked_ae.fit(featureMatrixTR, featureMatrixTR, batch_size=2048, epochs=100,
                    validation_split=0.2)

history = stacked_ae.fit(featureMatrixTR, featureMatrixTR,epochs=100, callbacks = [cp], validation_split=0.2)


loaded_model = tf.keras.models.load_model("EarlyFusion.h5")
loaded_model =loaded_model.layers[0]

# Report

def testing():
    loaded_model.predict(featureMatrix)



import timeit
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat("DNN.h5").st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)





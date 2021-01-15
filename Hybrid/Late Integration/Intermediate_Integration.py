
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
encoder = tf.keras.layers.Dense(units=50,input_shape=[100],name="ae_Features") (modelEncoder.output)
decoder = tf.keras.layers.Dense(units =419, input_shape=[50],name="ae_Output")(encoder)
model = keras.Model(inputs=[modelEncoder.input], outputs=[decoder])

# Compiling
model.compile(optimizer=tf.optimizers.Adam(),
            loss ="mse",
            )
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='stacked_Static_ae.png')
# Training

es = tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)
cp = tf.keras.callbacks.ModelCheckpoint("Combined_MultiSAE_2.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
history = model.fit([featureMatrixTRD,featureMatrixTRS], [featureMatrixTR], batch_size=500, epochs=1500, validation_split=0.2,callbacks=[cp, tensorboard_cb])


score = model.predict([featureMatrixD,featureMatrixS])

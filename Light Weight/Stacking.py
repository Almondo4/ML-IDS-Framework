import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# level 1
from sklearn.linear_model import LogisticRegression
#  Model Building
# import tensorflow as tf
# from tensorflow import keras
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(input_shape = 3, units=6, activation="relu",
#                               name="hiddenL_1"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
# model.add(keras.layers.Dense(units=6, activation="relu", name="hiddenL_3"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
# model.add(keras.layers.Dense(units=1, activation="softmax", name="outLayer"))
#
# # Compiling
# opt = tf.keras.optimizers.Adam()
# model.compile(optimizer=opt, loss ="binary_crossentropy", metrics =["accuracy"])

# Training

# level 0
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('etc', ExtraTreesClassifier(verbose=1,criterion= 'gini', n_estimators= 300)),
#     ('bg', BaggingClassifier()),
#
# ]
#
# from sklearn.ensemble import StackingClassifier
# clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(verbose=1))
# # Training
# clf = clf.fit(featureMatrixTR,labelVectorTR)
#
# # Testing
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import roc_auc_score
# y_pred2 = clf.predict(featureMatrix)
# print(classification_report(labelVector, y_pred2,digits=4))
# auc = roc_auc_score(labelVector, y_pred2)
# print('ROC AUC: %f' % auc)




# #######################################3

# compare ensemble to each baseline classifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingClassifier

# get a stacking ensemble of models

# define the base models
level0 = list()
level0.append(('rf', RandomForestClassifier(n_estimators=10, random_state=42)))
level0.append(('etc', ExtraTreesClassifier(verbose=1,criterion= 'gini', n_estimators= 300)))
level0.append(('bg', BaggingClassifier()))
level1 = LogisticRegression(verbose=1)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

def training():
    model.fit(featureMatrixTR, labelVectorTR)
model.fit(featureMatrixTR, labelVectorTR)
cv = model.fit(featureMatrixTR,labelVectorTR)
import pickle
filename = 'Stacked_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


def testing():
        loaded_model.predict(featureMatrix)

y_pred2 = cv.predict(featureMatrix)

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
import timeit
import os

print(classification_report(labelVector, y_pred2,digits=4))

auc = roc_auc_score(labelVector, y_pred2)
print('ROC AUC: %f' % auc)

train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat(filename).st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)


# evaluate a give model using cross-validation\







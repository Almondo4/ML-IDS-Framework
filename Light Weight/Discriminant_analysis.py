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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()

def training():
    model.fit(featureMatrixTR, labelVectorTR)

model.fit(featureMatrixTR, labelVectorTR)

import pickle
filename = 'DiscriminantAnalysis_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


def testing():
    loaded_model.predict(featureMatrix)


predictions = model.predict(featureMatrix)

## TESTING

from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(labelVector, predictions)
print(classification_report(labelVector, predictions,digits=4))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, predictions.round())
print('ROC AUC: %f' % auc)


import timeit
import os
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat(filename).st_size
# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)





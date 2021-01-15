import pandas as pd
import numpy as np

import os, psutil
pid = os.getpid()
ps = psutil.Process(pid)
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


# # Feature Extraction
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
# selector = SelectKBest(f_classif, k=100)
# selected_features = selector.fit_transform(featureMatrix, labelVector)
#
# print((-selector.scores_).argsort()[:])

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, verbose=3)

def training():
    clf.fit(featureMatrixTR, labelVectorTR)


clf.fit(featureMatrixTR, labelVectorTR)




import pickle
filename = 'RandomF_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



## TESTING
def testing ():
    loaded_model.predict(featureMatrix)

from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = loaded_model.predict(featureMatrix)

cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits =4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions.round())
print('ROC AUC: %f' % auc)


import timeit
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat('RandomF_model.sav').st_size

# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)



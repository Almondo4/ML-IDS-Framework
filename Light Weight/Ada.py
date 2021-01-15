import pandas as pd
import numpy as np
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




from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix as CM

clf = AdaBoostClassifier(n_estimators=150,algorithm="SAMME.R",)

def training():
    clf.fit(featureMatrixTR, labelVectorTR)

clf.fit(featureMatrixTR, labelVectorTR)

import pickle
filename = 'Ada_model.sav'
pickle.dump(clf, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


def testing():
    loaded_model.predict(featureMatrix)


y_pred2 = clf.predict(featureMatrix)


## TESTING
# predictions = cb.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
# XGB_predictions_Classes =model.predict_classes(test)
#
cm = confusion_matrix(labelVector, y_pred2)
print(classification_report(labelVector, y_pred2,digits=4))


auc = roc_auc_score(labelVector, y_pred2)
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




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


grid_param = {
    'n_estimators': [20,100, 300],
    'criterion': ['gini', 'entropy'],
    # 'max_depth': [2,4,12],
    'verbose': [1]
}
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix as CM

model = ExtraTreesClassifier(verbose=1,)
from sklearn.model_selection import GridSearchCV
gd_sr = GridSearchCV(estimator=model,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

gd_sr.fit(featureMatrixTR, labelVectorTR)
best_parameters = gd_sr.best_params_
print("Best Params: ",best_parameters)
best_result = gd_sr.best_score_
print("Best Results: ",best_result)

model = ExtraTreesClassifier(criterion=best_parameters["criterion"],n_estimators=best_parameters["n_estimators"],verbose=1,)

def training():
    model.fit(featureMatrixTR, labelVectorTR)

model=model.fit(featureMatrixTR,labelVectorTR)


import pickle
filename = 'EXTRAT_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


def testing():
    loaded_model.predict(featureMatrix)

y_pred2 = model.predict(featureMatrix)
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




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

DataTrain = pd.read_pickle("../../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../../Data/Dynamic_Testing.csv")


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

dynamicModel = keras.models.load_model("../../Hybrid/Late Integration/SAE_DLD_2.h5")
dynamicModel = dynamicModel.layers[0]


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


featureMatrixTR = dynamicModel.predict(featureMatrixTR)
featureMatrix =dynamicModel.predict(featureMatrix)


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)
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
y_pred2 = model.predict(featureMatrix)

print("Performance:",sum(y_pred2==labelVector)/len(labelVector))
print("Confusion Matrix:\n",CM(labelVector,y_pred2))



## TESTING

model = ExtraTreesClassifier(criterion= 'entropy', n_estimators= 100,verbose=1,)

model.fit(featureMatrixTR, labelVectorTR)

# Report
from sklearn.metrics import classification_report
model_predictions = model.predict(featureMatrix)
# cm = confusion_matrix(labelVector, model_predictions)
print(classification_report(labelVector, model_predictions.round(),digits =4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, model_predictions.round())
print('ROC AUC: %f' % auc)

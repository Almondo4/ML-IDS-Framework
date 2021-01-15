import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../../Data/AndMal_Zero_Train.csv")
DataTest = pd.read_pickle("../../../Data/AndMal_Zero_Test.csv")
DataZero = pd.read_pickle("../../../Data/AndMal_Zero_Day.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
featureMatrixZ = DataZero.iloc[:,:-1].values
labelVectorZ = DataZero.iloc[:,-1].values



#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)
featureMatrixZ = sc.fit_transform(featureMatrixZ)

labelVectorTR= labelVectorTR.astype(np.int32)
labelVector= labelVector.astype(np.int32)
labelVectorZ= labelVectorZ.astype(np.int32)

from tensorflow import keras
dynamicModel = keras.models.load_model("SAE_AndMAl_Zero.h5")
dynamicModel = dynamicModel.layers[0]


featureMatrixTR = dynamicModel.predict(featureMatrixTR)
featureMatrix =dynamicModel.predict(featureMatrix)
featureMatrixZ=dynamicModel.predict(featureMatrixZ)


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

#
# print("Performance:",sum(y_pred2==labelVector)/len(labelVector))
# print("Confusion Matrix:\n",CM(labelVector,y_pred2))
#


## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
model =ExtraTreesClassifier(n_estimators=best_parameters["n_estimators"],criterion=best_parameters["criterion"])
model=model.fit(featureMatrixTR,labelVectorTR)
RF_predictions = model.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions)
print('ROC AUC: %f' % auc)


# # Zero Day

print("############################################################ ZERO DAY")

from sklearn.metrics import classification_report
RF_predictions = model.predict(featureMatrixZ)
print(classification_report(labelVectorZ, RF_predictions,digits=4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVectorZ,RF_predictions)
print('ROC AUC: %f' % auc)

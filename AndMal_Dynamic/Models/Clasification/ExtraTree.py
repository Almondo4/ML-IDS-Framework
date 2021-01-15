import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/AndMal_2020_NoZD_Train.csv")
DataTest = pd.read_pickle("../../Data/AndMal_2020_NoZD_Test.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
# featureMatrixZ = DataZero.iloc[:,:-1].values
# labelVectorZ = DataZero.iloc[:,-1].values


#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)
# featureMatrixZ = sc.fit_transform(featureMatrixZ)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)
# labelVectorZ = labelencoder.fit_transform(labelVectorZ)


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



## TESTING

from sklearn.metrics import classification_report,confusion_matrix
model =ExtraTreesClassifier(n_estimators=best_parameters["n_estimators"],criterion=best_parameters["criterion"])
model=model.fit(featureMatrixTR,labelVectorTR)
RF_predictions = model.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
# auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo", labels=["0","1","2","3","4","5","6","7","8","9","10","11","12"] )
print('ROC AUC: %f' % auc)

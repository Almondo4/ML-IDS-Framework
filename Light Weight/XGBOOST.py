# import pandas as pd
# import numpy as np
# DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
# DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")
#
#
# featureMatrixTR = DataTrain.iloc[:,:-1].values
# labelVectorTR = DataTrain.iloc[:,-1].values
# featureMatrix = DataTest.iloc[:,:-1].values
# labelVector = DataTest.iloc[:,-1].values
#
#
# # #       3 Scaling the dataSet
# # from sklearn.preprocessing import StandardScaler
# # sc = StandardScaler()
# # featureMatrixTR = sc.fit_transform(featureMatrixTR)
# # featureMatrix = sc.fit_transform(featureMatrix)
#
# # from tensorflow.keras.utils import to_categorical
# # labelVector = to_categorical(labelVector)
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# labelVectorTR = labelencoder.fit_transform(labelVectorTR)
# labelVector = labelencoder.fit_transform(labelVector)
#
#
# # # Feature Extraction
# # from sklearn.feature_selection import SelectKBest
# # from sklearn.feature_selection import f_classif
# #
# # selector = SelectKBest(f_classif, k=100)
# # selected_features = selector.fit_transform(featureMatrix, labelVector)
# #
# # print((-selector.scores_).argsort()[:])
#
# import xgboost as xgb
# # train = xgb.DMatrix(featureMatrixTR, label=labelVectorTR)
# # test = xgb.DMatrix(featureMatrix, label=labelVector)
#
# # param = {
# #     "max_depth":420,
# #     "eta": 0.3,
# #     # "objective": "mutli:softmax",
# #     "num_class": 2,
# #     "verbosity": 3}
# # epochs = 300
#
# grid_param = {
#     'n_estimators': [20,100, 300],
#     "eta": [0.01,0.3,0.7,1],
#     'max_depth': [2,6,12],
#     "num_class": [2],
#     'verbose': [3]
# }
#
#
# model = xgb.XGBClassifier()
# from sklearn.model_selection import GridSearchCV
# gd_sr = GridSearchCV(estimator=model,
#                      param_grid=grid_param,
#                      scoring='accuracy',
#                      cv=5,
#                      n_jobs=-1)
#
# gd_sr.fit(featureMatrixTR, labelVectorTR)
# best_parameters = gd_sr.best_params_
# print("Best Params: ",best_parameters)
# best_result = gd_sr.best_score_
# print("Best Results: ",best_result)

# model = xgb.train(param, train, epochs)
# predictions = model.predict(test)
# y_pred2 = model.predict(featureMatrix)

# print("Performance:",sum(y_pred2==labelVector)/len(labelVector))
# print("Confusion Matrix:\n",CM(labelVector,y_pred2))


## TESTING

# from sklearn.metrics import classification_report,confusion_matrix
# XGB_predictions = model.predict(test)
# # XGB_predictions_Classes =model.predict_classes(test)
# #
# cm = confusion_matrix(labelVector, XGB_predictions)
# print(classification_report(labelVector, XGB_predictions))
#
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
#
# XGB_predictions = XGB_predictions[:, 0]
# # XGB_predictions_Classes = XGB_predictions_Classes[:, 0]
#
#
# print(classification_report(labelVector, XGB_predictions))
# # kappa = cohen_kappa_score(labelVector, XGB_predictions_Classes)
# # print('Cohens kappa: %f' % kappa)
# # ROC AUC
# auc = roc_auc_score(labelVector, XGB_predictions)
# print('ROC AUC: %f' % auc)
# # confusion matrix
# # matrix = confusion_matrix(labelVector, XGB_predictions_Classes)
# # print(matrix)







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


# # Feature Extraction
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
# selector = SelectKBest(f_classif, k=100)
# selected_features = selector.fit_transform(featureMatrix, labelVector)
#
# print((-selector.scores_).argsort()[:])

import xgboost as xgb
train = xgb.DMatrix(featureMatrixTR, label=labelVectorTR)
test = xgb.DMatrix(featureMatrix, label=labelVector)

param = {
    "max_depth":60,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 2,
    "verbosity": 3}
epochs = 20
def training():
    model = xgb.train(param, train, epochs)

model = xgb.train(param, train, epochs)

def testing():
    predictions = model.predict(test)

import pickle
filename = 'XGB_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(test)

## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = model.predict(test)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, RF_predictions.round())
print('ROC AUC: %f' % auc)


import timeit
import os
train_time = timeit.timeit(lambda: training(), number=10)
test_time = timeit.timeit(lambda: testing(), number=10)
x = os.stat('XGB_model.sav').st_size

# Light weight Tests"
print(" ########## Light weight Tests #################")
print("Model  Size", x)
print("TRaining Time",train_time)
print("Teststing Time",test_time)







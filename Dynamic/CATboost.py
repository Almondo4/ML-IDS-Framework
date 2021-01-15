import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../Data/Dynamic_Testing.csv")


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
import catboost as cb
cb = cb.CatBoostClassifier(depth=8,learning_rate=1,iterations=600,task_type="GPU",devices='0:1')

cb.fit(featureMatrixTR, labelVectorTR)



## TESTING
predictions = cb.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
# XGB_predictions_Classes =model.predict_classes(test)
#
cm = confusion_matrix(labelVector, predictions)
print(classification_report(labelVector, predictions,digits = 4))


auc = roc_auc_score(labelVector, predictions)
print('ROC AUC: %f' % auc)



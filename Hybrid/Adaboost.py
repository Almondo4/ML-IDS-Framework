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


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix as CM

clf = AdaBoostClassifier(n_estimators=150,algorithm="SAMME.R",)
clf.fit(featureMatrixTR,labelVectorTR)
y_pred2 = clf.predict(featureMatrix)

print("Performance:",sum(y_pred2==labelVector)/len(labelVector))
print("Confusion Matrix:\n",CM(labelVector,y_pred2))



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



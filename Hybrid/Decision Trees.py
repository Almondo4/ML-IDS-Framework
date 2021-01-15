import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)




from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_features=None)
clf=clf.fit(featureMatrixTR,labelVectorTR)


## TESTING
Y_pred = clf.predict(featureMatrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labelVector,Y_pred)
accuracy = float(cm.diagonal().sum())/len(labelVector)
print("\nAccuracy Of DTC For The Given Dataset : ", accuracy)

from sklearn.metrics import classification_report,confusion_matrix
tree_predictions = clf.predict(featureMatrix)
cm = confusion_matrix(labelVector, tree_predictions)
print(classification_report(labelVector, tree_predictions,digits =4))
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(labelVector, tree_predictions)
print('ROC AUC: %f' % auc)

import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/AndMal_2020_Train.csv")
DataTest = pd.read_pickle("../../Data/AndMal_2020_Test.csv")


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

from sklearn.naive_bayes import GaussianNB
## MODEL
# Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(featureMatrixTR, labelVectorTR)

#Predict the response for test dataset
y_pred = gnb.predict(featureMatrix)
# Model Accuracy, how often is the classifier correct?

## TESTING
# Y_pred = model.predict(featureMatrix)
from sklearn.metrics import classification_report,confusion_matrix
RF_predictions = gnb.predict(featureMatrix)
cm = confusion_matrix(labelVector, RF_predictions)
print(classification_report(labelVector, RF_predictions,digits=4))


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labelVector)
labelVector = lb.transform(labelVector)
RF_predictions = lb.transform(RF_predictions)

auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
# auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo", labels=["0","1","2","3","4","5","6","7","8","9","10","11","12"] )
print('ROC AUC: %f' % auc)


# from sklearn import metrics
# fpr, tpr, thresholds = metrics.roc_curve(labelVector, y_pred.round())
# print("TPR1: ",np.average(tpr))
# print("FPR1: ",np.average(fpr))
# cm = confusion_matrix(labelVector, y_pred.round())
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
#
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# print("TPR: ",np.average(TPR))
# # Specificity or true negative rate
# # TNR = TN/(TN+FP)
# # # Precision or positive predictive value
# # PPV = TP/(TP+FP)
# # print("PRICISION: ",PPV)
# # # Fall out or false positive rate
# FPR = FP/(FP+TN)
# print("FPR: ",np.average(FPR))
# # # False negative rate
# # FNR = FN/(TP+FN)
# # print("FNR: ",np.average(FNR))
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# print("ACC: ",ACC)
#
# Precision = TP/(TP+FP)
# print("Precision:",np.average(Precision))
# Recall = TP/(TP+FN)
# print("Recall", np.average(Recall))

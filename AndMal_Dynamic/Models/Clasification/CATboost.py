import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/AndMal_2020_Train.csv")
DataTest = pd.read_pickle("../../Data/AndMal_2020_Test.csv")



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
import catboost as cb
cb = cb.CatBoostClassifier(depth=8,learning_rate=1,iterations=600,task_type="GPU",devices='0:1')

cb.fit(featureMatrixTR, labelVectorTR)



## TESTING
from sklearn.metrics import classification_report,confusion_matrix

RF_predictions = cb.predict(featureMatrix)
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

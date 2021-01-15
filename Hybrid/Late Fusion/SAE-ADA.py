
# DATA
# # Dynamic
import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../../Data/Dynamic_Testing.csv")

featureMatrixTRD = DataTrain.iloc[:,:-1].values
labelVectorTRD = DataTrain.iloc[:,-1].values
featureMatrixD = DataTest.iloc[:,:-1].values
labelVectorD = DataTest.iloc[:,-1].values

# # Static

DataTrain = pd.read_pickle("../../Data/Static_Training.csv")
DataTest = pd.read_pickle("../../Data/Static_Testing.csv")

featureMatrixTRS = DataTrain.iloc[:,:-1].values
labelVectorTRS = DataTrain.iloc[:,-1].values
featureMatrixS = DataTest.iloc[:,:-1].values
labelVectorS = DataTest.iloc[:,-1].values
# # Hybrid
DataTrain = pd.read_pickle("../../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../../Data/Hybrid_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# Models
from tensorflow import keras


staticModel = keras.models.load_model("../Late Integration/SAE_DLS_2.h5")
dynamicModel = keras.models.load_model("../Late Integration/SAE_DLD_2.h5")

# Creating Datasets
# # Training
staticSet=staticModel.predict(featureMatrixTRS)
dynamicSet=dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)
# # Testing
staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=150,algorithm="SAMME.R",)
model.fit(DATATRaining,labelVectorTR)

# Report
from sklearn.metrics import classification_report
model_predictions = model.predict(DATATesting)
# cm = confusion_matrix(labelVector, model_predictions)
print(classification_report(labelVector, model_predictions.round(),digits =4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, model_predictions.round())
print('ROC AUC: %f' % auc)

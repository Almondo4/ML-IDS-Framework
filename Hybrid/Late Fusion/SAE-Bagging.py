
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

grid_param = {
    'n_estimators': [30],
    'max_features': [30],
    # 'max_depth': [2,4,12],
    'verbose': [1]
}
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import confusion_matrix as CM

model = BaggingClassifier()

# model.fit(featureMatrixTR, labelVectorTR)
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
model = BaggingClassifier(max_features=best_parameters["max_features"],n_estimators=best_parameters["n_estimators"],verbose=1,)
model=model.fit(DATATRaining,labelVectorTR)


# Report
from sklearn.metrics import classification_report
model_predictions = model.predict(DATATesting)
# cm = confusion_matrix(labelVector, model_predictions)
print(classification_report(labelVector, model_predictions.round(),digits =4))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labelVector, model_predictions.round())
print('ROC AUC: %f' % auc)
